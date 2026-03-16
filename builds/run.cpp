#include <ctype.h>
#include <exception>
#include <fcntl.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <cstdint>
#include <cstring>
#include <chrono>
#include <limits>
#include <unordered_map>
#include <vector>

auto constexpr VEC_SIZE_W  = 32;
auto constexpr VEC_SIZE_X  = 16;
auto constexpr VEC_SIZE_Y  = 16;
auto constexpr GROUP_BITS  = 8;
auto constexpr GROUP_SIZE  = 1 << GROUP_BITS;
auto constexpr PACKED_BYTES = VEC_SIZE_W * GROUP_BITS / 8;
auto constexpr MAX_N = 1408;
auto constexpr MAX_D = 32000;

struct PreQuantData {
    std::vector<uint8_t> packed;
    std::vector<float>   codebook;
    int padded_n;
    int padded_d;
};

static std::unordered_map<const float*, PreQuantData> g_prequant;

static std::vector<uint8_t> pack_indices_padded(
    const uint8_t* indices, int n, int d, int padded_n, int padded_d
) {
    int row_blocks = padded_n / VEC_SIZE_W;
    std::vector<uint8_t> packed((size_t)padded_d * row_blocks * PACKED_BYTES, 0);
    for (int r = 0; r < d; ++r) {
        for (int j = 0; j < row_blocks; ++j) {
            uint8_t* blk = packed.data() + ((size_t)r * row_blocks + j) * PACKED_BYTES;
            if constexpr (GROUP_BITS % 8 == 0) {
                constexpr int BYTES_PER_IDX = GROUP_BITS / 8;
                for (int e = 0; e < VEC_SIZE_W; ++e) {
                    int c = j * VEC_SIZE_W + e;
                    uint8_t idx_val = (c < n) ? indices[(size_t)r * n + c] : 0;
                    blk[e * BYTES_PER_IDX] = idx_val;
                }
            } else {
                for (int e = 0; e < VEC_SIZE_W; ++e) {
                    int c = j * VEC_SIZE_W + e;
                    uint8_t idx_val = (c < n) ? indices[(size_t)r * n + c] : 0;
                    int bit_pos  = e * GROUP_BITS;
                    int byte_idx = bit_pos / 8;
                    int bit_off  = bit_pos % 8;
                    blk[byte_idx] |= static_cast<uint8_t>(idx_val << bit_off);
                }
            }
        }
    }
    return packed;
}

static void quantize_w(
    const float*          w,
    int                   n,
    int                   d,
    int                   padded_n,
    int                   padded_d,
    std::vector<uint8_t>& packed,
    std::vector<float>&   codebook
) {
    float w_min =  std::numeric_limits<float>::max();
    float w_max = -std::numeric_limits<float>::max();
    for (int i = 0; i < d * n; ++i) {
        w_min = std::min(w_min, w[i]);
        w_max = std::max(w_max, w[i]);
    }

    codebook.resize(GROUP_SIZE);
    for (int k = 0; k < GROUP_SIZE; ++k)
        codebook[k] = w_min + (w_max - w_min) * (k + 0.5f) / GROUP_SIZE;

    int row_blocks = padded_n / VEC_SIZE_W;
    packed.assign((size_t)padded_d * row_blocks * PACKED_BYTES, 0);

    float step = (w_max > w_min) ? (w_max - w_min) / GROUP_SIZE : 1.0f;
    float inv_step = 1.0f / step;

    for (int r = 0; r < d; ++r) {
        for (int j = 0; j < row_blocks; ++j) {
            uint8_t* blk = packed.data() + ((size_t)r * row_blocks + j) * PACKED_BYTES;
            if constexpr (GROUP_BITS % 8 == 0) {
                constexpr int BYTES_PER_IDX = GROUP_BITS / 8;
                for (int e = 0; e < VEC_SIZE_W; ++e) {
                    int c = j * VEC_SIZE_W + e;
                    float val = (c < n) ? w[(size_t)r * n + c] : 0.0f;
                    int best = (int)((val - w_min) * inv_step);
                    if (best < 0) best = 0;
                    if (best >= GROUP_SIZE) best = GROUP_SIZE - 1;
                    blk[e * BYTES_PER_IDX] = static_cast<uint8_t>(best);
                }
            } else {
                for (int e = 0; e < VEC_SIZE_W; ++e) {
                    int c = j * VEC_SIZE_W + e;
                    float val = (c < n) ? w[(size_t)r * n + c] : 0.0f;
                    int best = (int)((val - w_min) * inv_step);
                    if (best < 0) best = 0;
                    if (best >= GROUP_SIZE) best = GROUP_SIZE - 1;
                    int bit_pos  = e * GROUP_BITS;
                    int byte_idx = bit_pos / 8;
                    int bit_off  = bit_pos % 8;
                    blk[byte_idx] |= static_cast<uint8_t>(best << bit_off);
                }
            }
        }
    }
}

struct WQuantBo {
    xrt::bo bo_w;
    xrt::bo bo_cb;
    WQuantBo() = default;
    WQuantBo(xrt::bo w, xrt::bo cb) : bo_w(std::move(w)), bo_cb(std::move(cb)) {}
};

class MatMulAccelerator {
private:
    xrt::device device;
    xrt::kernel krnl;
    xrt::bo bo_x;
    xrt::bo bo_y;
    xrt::run run;
    int max_n;
    int max_d;
    int padded_max_n;
    int padded_max_d;

    std::unordered_map<const float*, WQuantBo> w_bo_cache;

    float* bo_x_ptr;
    float* bo_y_ptr;

    std::chrono::high_resolution_clock::time_point t_start;
    double kernel_total_ms   = 0.0;
    long   kernel_call_count = 0;

public:
    MatMulAccelerator(const xrt::device& device, const xrt::uuid& uuid,
                      const std::string& cu_name, int max_n, int max_d)
        : device(device), max_n(max_n), max_d(max_d) {

        padded_max_n = ((max_n + VEC_SIZE_W - 1) / VEC_SIZE_W) * VEC_SIZE_W;
        padded_max_d = ((max_d + VEC_SIZE_Y - 1) / VEC_SIZE_Y) * VEC_SIZE_Y;

        krnl = xrt::kernel(device, uuid, cu_name);

        size_t x_sz = (size_t)padded_max_n * sizeof(float);
        size_t y_sz = (size_t)padded_max_d * sizeof(float);

        bo_x = xrt::bo(device, x_sz, xrt::bo::flags::cacheable, krnl.group_id(2));
        bo_y = xrt::bo(device, y_sz, xrt::bo::flags::cacheable, krnl.group_id(3));
        run  = xrt::run(krnl);

        bo_x_ptr = bo_x.map<float*>();
        bo_y_ptr = bo_y.map<float*>();
        std::memset(bo_x_ptr, 0, x_sz);
        std::memset(bo_y_ptr, 0, y_sz);
    }

    void preload_w(const float* w, int n, int d) {
        if (w_bo_cache.count(w)) return;

        int padded_n, padded_d;
        std::vector<uint8_t> packed;
        std::vector<float>   codebook;

        auto it = g_prequant.find(w);
        if (it != g_prequant.end()) {
            const PreQuantData& pqd = it->second;
            padded_n = pqd.padded_n;
            padded_d = pqd.padded_d;
            packed   = pqd.packed;
            codebook = pqd.codebook;
        } else {
            padded_n = ((n + VEC_SIZE_W - 1) / VEC_SIZE_W) * VEC_SIZE_W;
            padded_d = ((d + VEC_SIZE_Y - 1) / VEC_SIZE_Y) * VEC_SIZE_Y;
            quantize_w(w, n, d, padded_n, padded_d, packed, codebook);
        }

        size_t w_bytes  = packed.size();
        size_t cb_bytes = (size_t)GROUP_SIZE * sizeof(float);

        xrt::bo bo_w(device, w_bytes, krnl.group_id(0));
        bo_w.write(packed.data(), w_bytes, 0);
        bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE, w_bytes, 0);

        xrt::bo bo_cb(device, cb_bytes, krnl.group_id(1));
        bo_cb.write(codebook.data(), cb_bytes, 0);
        bo_cb.sync(XCL_BO_SYNC_BO_TO_DEVICE, cb_bytes, 0);

        w_bo_cache.emplace(w, WQuantBo(std::move(bo_w), std::move(bo_cb)));
    }

    void start_task(const float* w, const float* x, int n, int d,
                    bool skip_x_sync = false) {
        if (n > max_n || d > max_d) {
            std::cerr << "Error: Size exceeds max buffer" << std::endl;
            return;
        }

        auto it = w_bo_cache.find(w);
        if (it == w_bo_cache.end()) {
            std::cerr << "[ERROR] Weight not preloaded. Call preload_w() before start_task().\n";
            exit(EXIT_FAILURE);
        }

        int padded_n   = ((n + VEC_SIZE_W - 1) / VEC_SIZE_W) * VEC_SIZE_W;
        int padded_d   = ((d + VEC_SIZE_Y - 1) / VEC_SIZE_Y) * VEC_SIZE_Y;
        size_t x_bytes = (size_t)padded_n * sizeof(float);

        std::memset(bo_x_ptr + n, 0, (padded_n - n) * sizeof(float));
        if (!skip_x_sync) {
            std::memcpy(bo_x_ptr, x, n * sizeof(float));
            bo_x.sync(XCL_BO_SYNC_BO_TO_DEVICE, x_bytes, 0);
        }

        run.set_arg(0, it->second.bo_w);
        run.set_arg(1, it->second.bo_cb);
        run.set_arg(2, bo_x);
        run.set_arg(3, bo_y);
        run.set_arg(4, padded_n);
        run.set_arg(5, padded_d);

        run.start();
        t_start = std::chrono::high_resolution_clock::now();
    }

    void wait_task(float* y, int d) {
        run.wait();
        kernel_total_ms += std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_start).count();
        ++kernel_call_count;

        int padded_d   = ((d + VEC_SIZE_Y - 1) / VEC_SIZE_Y) * VEC_SIZE_Y;
        size_t y_bytes = (size_t)padded_d * sizeof(float);

        bo_y.sync(XCL_BO_SYNC_BO_FROM_DEVICE, y_bytes, 0);
        std::memcpy(y, bo_y_ptr, d * sizeof(float));
    }

    void print_stats(const std::string& name) const {
        if (kernel_call_count == 0) return;
        fprintf(stderr, "[FPGA] %-35s calls=%ld  total=%.2f ms  avg=%.4f ms\n",
                name.c_str(), kernel_call_count,
                kernel_total_ms, kernel_total_ms / kernel_call_count);
    }
};


class FpgaManager {
public:
    xrt::device device;
    xrt::uuid uuid;
    std::vector<MatMulAccelerator*> accels;

    FpgaManager(const std::string& xclbin_path) {
        device = xrt::device(0);
        uuid   = device.load_xclbin(xclbin_path);

        accels.push_back(new MatMulAccelerator(
            device, uuid, "matmul_quantize_kernel:{matmul_quantize_kernel_1}", MAX_N, MAX_D));
    }

    ~FpgaManager() {
        fprintf(stderr, "\n[FPGA] ---- kernel execution stats ----\n");
        for (size_t i = 0; i < accels.size(); ++i)
            accels[i]->print_stats("matmul_quantize_kernel[" + std::to_string(i) + "]");
        for (auto accel : accels) delete accel;
    }

    void preload_w(int id, const float* w, int n, int d) {
        accels[id]->preload_w(w, n, d);
    }
};

static FpgaManager* get_manager() {
    try {
        static FpgaManager manager("binary_container_1.xclbin");
        return &manager;
    } catch (const std::exception& e) {
        std::cerr << "\n[XRT Init Error] " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}

extern "C" {
void matmul_preload_w(int id, const float *w, int n, int d) {
  try {
    get_manager()->preload_w(id, w, n, d);
  } catch (...) {
    std::cerr << "[ERROR] Exception in matmul_preload_w\n";
    exit(EXIT_FAILURE);
  }
}

void matmul_async_start(int id, const float *w, const float *x, int n, int d, bool skip_x_sync = false) {
  try {
    auto mgr = get_manager();
    mgr->accels[id]->start_task(w, x, n, d, skip_x_sync);
  } catch (...) {
    std::cerr << "[ERROR] Exception in start_task\n";
    exit(EXIT_FAILURE);
  }
}

void matmul_async_wait(int id, float *y, int d) {
  try {
    get_manager()->accels[id]->wait_task(y, d);
  } catch (...) {
    std::cerr << "[ERROR] Exception in wait_task\n";
    exit(EXIT_FAILURE);
  }
}

void matmul(float *y, const float *w, const float *x, int n, int d) {
  matmul_async_start(0, w, x, n, d);
  matmul_async_wait(0, y, d);
}
}

typedef struct {
  int dim;
  int hidden_dim;
  int n_layers;
  int n_heads;
  int n_kv_heads;
  int vocab_size;
  int seq_len;
} Config;

typedef struct {
  float *token_embedding_table;
  float *rms_att_weight;
  float *rms_ffn_weight;
  float *wq;
  float *wk;
  float *wv;
  float *wo;
  float *w1;
  float *w2;
  float *w3;
  float *rms_final_weight;
  float *wcls;
} TransformerWeights;

typedef struct {
  float *x;
  float *xb;
  float *xb2;
  float *hb;
  float *hb2;
  float *q;
  float *k;
  float *v;
  float *att;
  float *logits;
  float *key_cache;
  float *value_cache;
} RunState;

typedef struct {
  Config config;
  TransformerWeights weights;
  RunState state;
  int fd;
  float *data;
  ssize_t file_size;
} Transformer;

void malloc_run_state(RunState *s, Config *p) {
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  s->x = (float *)calloc(p->dim + 16, sizeof(float));
  s->xb = (float *)calloc(p->dim + 16, sizeof(float));
  s->xb2 = (float *)calloc(p->dim + 16, sizeof(float));
  s->hb = (float *)calloc(p->hidden_dim + 16, sizeof(float));
  s->hb2 = (float *)calloc(p->hidden_dim + 16, sizeof(float));
  s->q = (float *)calloc(p->dim + 16, sizeof(float));
  s->key_cache =
      (float *)calloc(p->n_layers * p->seq_len * kv_dim + 16, sizeof(float));
  s->value_cache =
      (float *)calloc(p->n_layers * p->seq_len * kv_dim + 16, sizeof(float));
  s->att = (float *)calloc(p->n_heads * p->seq_len + 16, sizeof(float));
  s->logits = (float *)calloc(p->vocab_size + 16, sizeof(float));
  if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q ||
      !s->key_cache || !s->value_cache || !s->att || !s->logits) {
    fprintf(stderr, "malloc failed!\n");
    exit(EXIT_FAILURE);
  }
}

void free_run_state(RunState *s) {
  free(s->x);
  free(s->xb);
  free(s->xb2);
  free(s->hb);
  free(s->hb2);
  free(s->q);
  free(s->att);
  free(s->logits);
  free(s->key_cache);
  free(s->value_cache);
}

void memory_map_weights(TransformerWeights *w, Config *p, float *ptr,
                        int shared_weights) {
  int head_size = p->dim / p->n_heads;
  unsigned long long n_layers = p->n_layers;
  w->token_embedding_table = ptr;
  ptr += p->vocab_size * p->dim;
  w->rms_att_weight = ptr;
  ptr += n_layers * p->dim;
  w->wq = ptr;
  ptr += n_layers * p->dim * (p->n_heads * head_size);
  w->wk = ptr;
  ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
  w->wv = ptr;
  ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
  w->wo = ptr;
  ptr += n_layers * (p->n_heads * head_size) * p->dim;
  w->rms_ffn_weight = ptr;
  ptr += n_layers * p->dim;
  w->w1 = ptr;
  ptr += n_layers * p->dim * p->hidden_dim;
  w->w2 = ptr;
  ptr += n_layers * p->hidden_dim * p->dim;
  w->w3 = ptr;
  ptr += n_layers * p->dim * p->hidden_dim;
  w->rms_final_weight = ptr;
  ptr += p->dim;
  ptr += p->seq_len * head_size / 2;
  ptr += p->seq_len * head_size / 2;
  w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(char *checkpoint, Config *config,
                     TransformerWeights *weights, int *fd, float **data,
                     ssize_t *file_size) {
  FILE *file = fopen(checkpoint, "rb");
  if (!file) {
    fprintf(stderr, "Couldn't open file %s\n", checkpoint);
    exit(EXIT_FAILURE);
  }
  if (fread(config, sizeof(Config), 1, file) != 1) {
    exit(EXIT_FAILURE);
  }
  int shared_weights = config->vocab_size > 0 ? 1 : 0;
  config->vocab_size = abs(config->vocab_size);
  fseek(file, 0, SEEK_END);
  *file_size = ftell(file);
  fclose(file);
  *fd = open(checkpoint, O_RDONLY);
  if (*fd == -1) {
    fprintf(stderr, "open failed!\n");
    exit(EXIT_FAILURE);
  }
  *data = (float *)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
  if (*data == MAP_FAILED) {
    fprintf(stderr, "mmap failed!\n");
    exit(EXIT_FAILURE);
  }
  float *weights_ptr = *data + sizeof(Config) / sizeof(float);
  memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer *t, char *checkpoint_path) {
  read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data,
                  &t->file_size);
  malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer *t) {
  if (t->data != MAP_FAILED) {
    munmap(t->data, t->file_size);
  }
  if (t->fd != -1) {
    close(t->fd);
  }
  free_run_state(&t->state);
}

static void load_quantized_weights(
    const char* quant_path, const char* cb_path, const Transformer* t
) {
    const Config* p = &t->config;
    const TransformerWeights* w = &t->weights;

    int dim        = p->dim;
    int kv_dim     = (p->dim * p->n_kv_heads) / p->n_heads;
    int hidden_dim = p->hidden_dim;
    int n_layers   = p->n_layers;
    int vocab_size = p->vocab_size;

    FILE* qf = fopen(quant_path, "rb");
    if (!qf) {
        fprintf(stderr, "Cannot open quant file: %s\n", quant_path);
        exit(EXIT_FAILURE);
    }
    FILE* cf = fopen(cb_path, "rb");
    if (!cf) {
        fprintf(stderr, "Cannot open codebook file: %s\n", cb_path);
        fclose(qf);
        exit(EXIT_FAILURE);
    }

    Config qcfg;
    if (fread(&qcfg, sizeof(Config), 1, qf) != 1) {
        fprintf(stderr, "Failed to read quant file header\n"); exit(EXIT_FAILURE);
    }
    int n_clusters_q;
    if (fread(&n_clusters_q, sizeof(int), 1, qf) != 1) {
        fprintf(stderr, "Failed to read quant n_clusters\n"); exit(EXIT_FAILURE);
    }

    Config ccfg;
    if (fread(&ccfg, sizeof(Config), 1, cf) != 1) {
        fprintf(stderr, "Failed to read codebook file header\n"); exit(EXIT_FAILURE);
    }
    int n_clusters_c;
    if (fread(&n_clusters_c, sizeof(int), 1, cf) != 1) {
        fprintf(stderr, "Failed to read codebook n_clusters\n"); exit(EXIT_FAILURE);
    }

    if (n_clusters_q != GROUP_SIZE || n_clusters_c != GROUP_SIZE) {
        fprintf(stderr, "n_clusters mismatch: quant=%d, codebook=%d, kernel=%d\n",
                n_clusters_q, n_clusters_c, GROUP_SIZE);
        fclose(qf);
        fclose(cf);
        exit(EXIT_FAILURE);
    }

    off_t skip = ((long long)vocab_size * dim + (long long)n_layers * dim) * sizeof(float);
    if (fseeko(qf, skip, SEEK_CUR) != 0) {
        fprintf(stderr, "fseeko failed on quant file\n"); exit(EXIT_FAILURE);
    }

    struct WDesc {
        float* base;
        int n, d;
        long long layer_stride;
    };

    std::vector<WDesc> descs = {
        {w->wq, dim,        dim,        (long long)dim * dim},
        {w->wk, dim,        kv_dim,     (long long)dim * kv_dim},
        {w->wv, dim,        kv_dim,     (long long)dim * kv_dim},
        {w->wo, dim,        dim,        (long long)dim * dim},
        {w->w1, dim,        hidden_dim, (long long)dim * hidden_dim},
        {w->w2, hidden_dim, dim,        (long long)hidden_dim * dim},
        {w->w3, dim,        hidden_dim, (long long)dim * hidden_dim},
    };

    for (int wi = 0; wi < (int)descs.size(); ++wi) {
        const WDesc& desc = descs[wi];
        int n   = desc.n;
        int d   = desc.d;
        long long sz = (long long)n * d;

        std::vector<std::vector<float>> layer_cbs(n_layers, std::vector<float>(GROUP_SIZE));
        for (int l = 0; l < n_layers; ++l) {
            if (fread(layer_cbs[l].data(), sizeof(float), GROUP_SIZE, cf) != (size_t)GROUP_SIZE) {
                fprintf(stderr, "Failed to read codebook for weight %d layer %d\n", wi, l);
                exit(EXIT_FAILURE);
            }
        }

        std::vector<std::vector<uint8_t>> layer_indices(n_layers, std::vector<uint8_t>(sz));
        for (int l = 0; l < n_layers; ++l) {
            if (fread(layer_indices[l].data(), sizeof(uint8_t), sz, qf) != (size_t)sz) {
                fprintf(stderr, "Failed to read indices for weight %d layer %d\n", wi, l);
                exit(EXIT_FAILURE);
            }
        }

        if (wi == 3) {
            off_t rms_skip = (long long)n_layers * dim * sizeof(float);
            if (fseeko(qf, rms_skip, SEEK_CUR) != 0) {
                fprintf(stderr, "fseeko failed skipping rms_ffn_weight\n"); exit(EXIT_FAILURE);
            }
        }

        int padded_n = ((n + VEC_SIZE_W - 1) / VEC_SIZE_W) * VEC_SIZE_W;
        int padded_d = ((d + VEC_SIZE_Y - 1) / VEC_SIZE_Y) * VEC_SIZE_Y;

        for (int l = 0; l < n_layers; ++l) {
            const float* key = desc.base + (long long)l * desc.layer_stride;
            PreQuantData pqd;
            pqd.padded_n = padded_n;
            pqd.padded_d = padded_d;
            pqd.packed   = pack_indices_padded(layer_indices[l].data(), n, d, padded_n, padded_d);
            pqd.codebook = layer_cbs[l];
            g_prequant.emplace(key, std::move(pqd));
        }
    }

    fclose(qf);
    fclose(cf);

    fprintf(stderr, "Loaded pre-quantized weights: %s, %s\n", quant_path, cb_path);
}

void preload_transformer_weights(Transformer* t,
                                 const char* quant_path = nullptr,
                                 const char* cb_path    = nullptr) {
    Config* p = &t->config;
    TransformerWeights* w = &t->weights;
    int dim        = p->dim;
    int kv_dim     = (p->dim * p->n_kv_heads) / p->n_heads;
    int hidden_dim = p->hidden_dim;

    if (quant_path && cb_path) {
        load_quantized_weights(quant_path, cb_path, t);
    }

    fprintf(stderr, "Preloading transformer weights to FPGA...\n");

    for (int l = 0; l < p->n_layers; l++) {
        long long ll = l;
        matmul_preload_w(0, w->wq + ll*dim*dim,         dim,        dim       );
        matmul_preload_w(0, w->wk + ll*dim*kv_dim,      dim,        kv_dim    );
        matmul_preload_w(0, w->wv + ll*dim*kv_dim,      dim,        kv_dim    );
        matmul_preload_w(0, w->wo + ll*dim*dim,         dim,        dim       );
        matmul_preload_w(0, w->w2 + ll*hidden_dim*dim,  hidden_dim, dim       );
        matmul_preload_w(0, w->w1 + ll*dim*hidden_dim,  dim,        hidden_dim);
        matmul_preload_w(0, w->w3 + ll*dim*hidden_dim,  dim,        hidden_dim);
    }

    matmul_preload_w(0, w->wcls, p->dim, p->vocab_size);

    fprintf(stderr, "Weight preload complete.\n");
}

void rmsnorm(float *o, float *x, float *weight, int size) {
  float ss = 0.0f;
  for (int j = 0; j < size; j++) {
    ss += x[j] * x[j];
  }
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);
  for (int j = 0; j < size; j++) {
    o[j] = weight[j] * (ss * x[j]);
  }
}

void softmax(float *x, int size) {
  float max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  for (int i = 0; i < size; i++) {
    x[i] /= sum;
  }
}

float *forward(Transformer *transformer, int token, int pos) {
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;
  float *x = s->x;
  int dim = p->dim;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  int kv_mul = p->n_heads / p->n_kv_heads;
  int hidden_dim = p->hidden_dim;
  int head_size = dim / p->n_heads;

  float *content_row = w->token_embedding_table + token * dim;
  memcpy(x, content_row, dim * sizeof(*x));

  float scale = 1.0f / sqrtf(head_size);

  for (unsigned long long l = 0; l < p->n_layers; l++) {
    rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

    int loff = l * p->seq_len * kv_dim;
    s->k = s->key_cache + loff + pos * kv_dim;
    s->v = s->value_cache + loff + pos * kv_dim;
    matmul_async_start(0, w->wq + l * dim * dim, s->xb, dim, dim, false);
    matmul_async_wait(0, s->q, dim);
    matmul_async_start(0, w->wk + l * dim * kv_dim, s->xb, dim, kv_dim, true);
    matmul_async_wait(0, s->k, kv_dim);
    matmul_async_start(0, w->wv + l * dim * kv_dim, s->xb, dim, kv_dim, true);
    matmul_async_wait(0, s->v, kv_dim);

    for (int i = 0; i < dim; i += 2) {
      int head_dim = i % head_size;
      float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
      float val = pos * freq;
      float fcr = cosf(val);
      float fci = sinf(val);
      int rotn = i < kv_dim ? 2 : 1;
      for (int v = 0; v < rotn; v++) {
        float *vec = v == 0 ? s->q : s->k;
        float v0 = vec[i];
        float v1 = vec[i + 1];
        vec[i] = v0 * fcr - v1 * fci;
        vec[i + 1] = v0 * fci + v1 * fcr;
      }
    }

    int h;
#pragma omp parallel for private(h)
    for (h = 0; h < p->n_heads; h++) {
      float *q = s->q + h * head_size;
      float *att = s->att + h * p->seq_len;
      for (int t = 0; t <= pos; t++) {
        float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) {
          score += q[i] * k[i];
        }
        score *= scale;
        att[t] = score;
      }

      softmax(att, pos + 1);

      float *xb = s->xb + h * head_size;
      memset(xb, 0, head_size * sizeof(float));
      for (int t = 0; t <= pos; t++) {
        float *v =
            s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        float a = att[t];
        for (int i = 0; i < head_size; i++) {
          xb[i] += a * v[i];
        }
      }
    }

    matmul_async_start(0, w->wo + l * dim * dim, s->xb, dim, dim);
    matmul_async_wait(0, s->xb2, dim);

    for (int i = 0; i < dim; i++) {
      x[i] += s->xb2[i];
    }

    rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

    matmul_async_start(0, w->w1 + l * dim * hidden_dim, s->xb, dim, hidden_dim);
    matmul_async_wait(0, s->hb, hidden_dim);
    matmul_async_start(0, w->w3 + l * dim * hidden_dim, s->xb, dim, hidden_dim);
    matmul_async_wait(0, s->hb2, hidden_dim);

    for (int i = 0; i < hidden_dim; i++) {
      float val = s->hb[i];
      val *= (1.0f / (1.0f + expf(-val)));
      val *= s->hb2[i];
      s->hb[i] = val;
    }

    matmul_async_start(0, w->w2 + l * dim * hidden_dim, s->hb, hidden_dim, dim);
    matmul_async_wait(0, s->xb, dim);

    for (int i = 0; i < dim; i++) {
      x[i] += s->xb[i];
    }
  }

  rmsnorm(x, x, w->rms_final_weight, dim);

  matmul_async_start(0, w->wcls, x, p->dim, p->vocab_size);
  matmul_async_wait(0, s->logits, p->vocab_size);

  return s->logits;
}

typedef struct {
  char *str;
  int id;
} TokenIndex;

typedef struct {
  char **vocab;
  float *vocab_scores;
  TokenIndex *sorted_vocab;
  int vocab_size;
  unsigned int max_token_length;
  unsigned char byte_pieces[512];
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
  return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

void build_tokenizer(Tokenizer *t, char *tokenizer_path, int vocab_size) {
  t->vocab_size = vocab_size;
  t->vocab = (char **)malloc(vocab_size * sizeof(char *));
  t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));
  t->sorted_vocab = NULL;
  for (int i = 0; i < 256; i++) {
    t->byte_pieces[i * 2] = (unsigned char)i;
    t->byte_pieces[i * 2 + 1] = '\0';
  }
  FILE *file = fopen(tokenizer_path, "rb");
  if (!file) {
    fprintf(stderr, "couldn't load %s\n", tokenizer_path);
    exit(EXIT_FAILURE);
  }
  if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) {
    fprintf(stderr, "failed read\n");
    exit(EXIT_FAILURE);
  }
  int len;
  for (int i = 0; i < vocab_size; i++) {
    if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    if (fread(&len, sizeof(int), 1, file) != 1) {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    t->vocab[i] = (char *)malloc(len + 1);
    if (fread(t->vocab[i], len, 1, file) != 1) {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    t->vocab[i][len] = '\0';
  }
  fclose(file);
}

void free_tokenizer(Tokenizer *t) {
  for (int i = 0; i < t->vocab_size; i++) {
    free(t->vocab[i]);
  }
  free(t->vocab);
  free(t->vocab_scores);
  free(t->sorted_vocab);
}

char *decode(Tokenizer *t, int prev_token, int token) {
  char *piece = t->vocab[token];
  if (prev_token == 1 && piece[0] == ' ') {
    piece++;
  }
  unsigned char byte_val;
  if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
    piece = (char *)t->byte_pieces + byte_val * 2;
  }
  return piece;
}

void safe_printf(char *piece) {
  if (piece == NULL) { return; }
  if (piece[0] == '\0') { return; }
  if (piece[1] == '\0') {
    unsigned char byte_val = piece[0];
    if (!(isprint(byte_val) || isspace(byte_val))) { return; }
  }
  printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
  TokenIndex tok = {.str = str};
  TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, vocab_size,
                                          sizeof(TokenIndex), compare_tokens);
  return res != NULL ? res->id : -1;
}

void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens,
            int *n_tokens) {
  if (text == NULL) {
    fprintf(stderr, "cannot encode NULL text\n");
    exit(EXIT_FAILURE);
  }
  if (t->sorted_vocab == NULL) {
    t->sorted_vocab = (TokenIndex *)malloc(t->vocab_size * sizeof(TokenIndex));
    for (int i = 0; i < t->vocab_size; i++) {
      t->sorted_vocab[i].str = t->vocab[i];
      t->sorted_vocab[i].id = i;
    }
    qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
  }
  char *str_buffer =
      (char *)malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
  size_t str_len = 0;
  *n_tokens = 0;
  if (bos) tokens[(*n_tokens)++] = 1;
  if (text[0] != '\0') {
    int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
    tokens[(*n_tokens)++] = dummy_prefix;
  }
  for (char *c = text; *c != '\0'; c++) {
    if ((*c & 0xC0) != 0x80) { str_len = 0; }
    str_buffer[str_len++] = *c;
    str_buffer[str_len] = '\0';
    if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) { continue; }
    int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
    if (id != -1) {
      tokens[(*n_tokens)++] = id;
    } else {
      for (int i = 0; i < (int)str_len; i++) {
        tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
      }
    }
    str_len = 0;
  }
  while (1) {
    float best_score = -1e10;
    int best_id = -1;
    int best_idx = -1;
    for (int i = 0; i < (*n_tokens - 1); i++) {
      sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
      int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
      if (id != -1 && t->vocab_scores[id] > best_score) {
        best_score = t->vocab_scores[id];
        best_id = id;
        best_idx = i;
      }
    }
    if (best_idx == -1) { break; }
    tokens[best_idx] = best_id;
    for (int i = best_idx + 1; i < (*n_tokens - 1); i++) {
      tokens[i] = tokens[i + 1];
    }
    (*n_tokens)--;
  }
  if (eos) tokens[(*n_tokens)++] = 2;
  free(str_buffer);
}

typedef struct {
  float prob;
  int index;
} ProbIndex;

typedef struct {
  int vocab_size;
  ProbIndex *probindex;
  float temperature;
  float topp;
  unsigned long long rng_state;
} Sampler;

int sample_argmax(float *probabilities, int n) {
  int max_i = 0;
  float max_p = probabilities[0];
  for (int i = 1; i < n; i++) {
    if (probabilities[i] > max_p) { max_i = i; max_p = probabilities[i]; }
  }
  return max_i;
}

int sample_mult(float *probabilities, int n, float coin) {
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += probabilities[i];
    if (coin < cdf) { return i; }
  }
  return n - 1;
}

int compare(const void *a, const void *b) {
  ProbIndex *a_ = (ProbIndex *)a;
  ProbIndex *b_ = (ProbIndex *)b;
  if (a_->prob > b_->prob) return -1;
  if (a_->prob < b_->prob) return 1;
  return 0;
}

int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex,
                float coin) {
  int n0 = 0;
  const float cutoff = (1.0f - topp) / (n - 1);
  for (int i = 0; i < n; i++) {
    if (probabilities[i] >= cutoff) {
      probindex[n0].index = i;
      probindex[n0].prob = probabilities[i];
      n0++;
    }
  }
  qsort(probindex, n0, sizeof(ProbIndex), compare);
  float cumulative_prob = 0.0f;
  int last_idx = n0 - 1;
  for (int i = 0; i < n0; i++) {
    cumulative_prob += probindex[i].prob;
    if (cumulative_prob > topp) { last_idx = i; break; }
  }
  float r = coin * cumulative_prob;
  float cdf = 0.0f;
  for (int i = 0; i <= last_idx; i++) {
    cdf += probindex[i].prob;
    if (r < cdf) { return probindex[i].index; }
  }
  return probindex[last_idx].index;
}

void build_sampler(Sampler *sampler, int vocab_size, float temperature,
                   float topp, unsigned long long rng_seed) {
  sampler->vocab_size = vocab_size;
  sampler->temperature = temperature;
  sampler->topp = topp;
  sampler->rng_state = rng_seed;
  sampler->probindex =
      (ProbIndex *)malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler *sampler) { free(sampler->probindex); }

unsigned int random_u32(unsigned long long *state) {
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) {
  return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler *sampler, float *logits) {
  int next;
  if (sampler->temperature == 0.0f) {
    next = sample_argmax(logits, sampler->vocab_size);
  } else {
    for (int q = 0; q < sampler->vocab_size; q++) {
      logits[q] /= sampler->temperature;
    }
    softmax(logits, sampler->vocab_size);
    float coin = random_f32(&sampler->rng_state);
    if (sampler->topp <= 0 || sampler->topp >= 1) {
      next = sample_mult(logits, sampler->vocab_size, coin);
    } else {
      next = sample_topp(logits, sampler->vocab_size, sampler->topp,
                         sampler->probindex, coin);
    }
  }
  return next;
}

long time_in_ms() {
  struct timespec time;
  clock_gettime(CLOCK_REALTIME, &time);
  return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
              char *prompt, int steps) {
  char *empty_prompt = "";
  if (prompt == NULL) { prompt = empty_prompt; }

  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int));
  encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
  if (num_prompt_tokens < 1) {
    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  long start = 0;
  int next;
  int token = prompt_tokens[0];
  int pos = 0;
  while (pos < steps) {
    float *logits = forward(transformer, token, pos);
    if (pos < num_prompt_tokens - 1) {
      next = prompt_tokens[pos + 1];
    } else {
      next = sample(sampler, logits);
    }
    pos++;
    if (next == 1) { break; }
    char *piece = decode(tokenizer, token, next);
    safe_printf(piece);
    fflush(stdout);
    token = next;
    if (start == 0) { start = time_in_ms(); }
  }
  printf("\n");
  if (pos > 1) {
    long end = time_in_ms();
    fprintf(stderr, "achieved tok/s: %f\n",
            (pos - 1) / (double)(end - start) * 1000);
  }
  free(prompt_tokens);
}

void read_stdin(const char *guide, char *buffer, size_t bufsize) {
  printf("%s", guide);
  if (fgets(buffer, bufsize, stdin) != NULL) {
    size_t len = strlen(buffer);
    if (len > 0 && buffer[len - 1] == '\n') { buffer[len - 1] = '\0'; }
  }
}

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {
  char system_prompt[512];
  char user_prompt[512];
  char rendered_prompt[1152];
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc(1152 * sizeof(int));
  int user_idx;
  int8_t user_turn = 1;
  int next;
  int token;
  int prev_token;
  int pos = 0;
  while (pos < steps) {
    if (user_turn) {
      if (pos == 0) {
        if (cli_system_prompt == NULL) {
          read_stdin("Enter system prompt (optional): ", system_prompt,
                     sizeof(system_prompt));
        } else {
          strcpy(system_prompt, cli_system_prompt);
        }
      }
      if (pos == 0 && cli_user_prompt != NULL) {
        strcpy(user_prompt, cli_user_prompt);
      } else {
        read_stdin("User: ", user_prompt, sizeof(user_prompt));
      }
      if (pos == 0 && system_prompt[0] != '\0') {
        char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
        sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
      } else {
        char user_template[] = "[INST] %s [/INST]";
        sprintf(rendered_prompt, user_template, user_prompt);
      }
      encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
      user_idx = 0;
      user_turn = 0;
      printf("Assistant: ");
    }
    if (user_idx < num_prompt_tokens) {
      token = prompt_tokens[user_idx++];
    } else {
      token = next;
    }
    if (token == 2) { user_turn = 1; }
    float *logits = forward(transformer, token, pos);
    next = sample(sampler, logits);
    pos++;
    if (user_idx >= num_prompt_tokens && next != 2) {
      char *piece = decode(tokenizer, token, next);
      safe_printf(piece);
      fflush(stdout);
    }
    if (next == 2) { printf("\n"); }
  }
  printf("\n");
  free(prompt_tokens);
}

#ifndef TESTING

void error_usage() {
  fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
  fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
  fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
  fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
  fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
  fprintf(stderr, "  -i <string> input prompt\n");
  fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
  fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
  fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
  fprintf(stderr, "  -q <string> path to pre-quantized weight bin (quantize.py output)\n");
  fprintf(stderr, "  -c <string> path to codebook bin (quantize.py output)\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
  char *checkpoint_path = NULL;
  char *tokenizer_path  = "tokenizer.bin";
  float temperature     = 1.0f;
  float topp            = 0.9f;
  int steps             = 256;
  char *prompt          = NULL;
  unsigned long long rng_seed = 0;
  char *mode            = "generate";
  char *system_prompt   = NULL;
  char *quant_path      = NULL;
  char *cb_path         = NULL;

  if (argc >= 2) {
    checkpoint_path = argv[1];
  } else {
    error_usage();
  }

  for (int i = 2; i < argc; i += 2) {
    if (i + 1 >= argc) { error_usage(); }
    if (argv[i][0] != '-') { error_usage(); }
    if (strlen(argv[i]) != 2) { error_usage(); }

    if      (argv[i][1] == 't') { temperature   = atof(argv[i + 1]); }
    else if (argv[i][1] == 'p') { topp           = atof(argv[i + 1]); }
    else if (argv[i][1] == 's') { rng_seed       = atoi(argv[i + 1]); }
    else if (argv[i][1] == 'n') { steps          = atoi(argv[i + 1]); }
    else if (argv[i][1] == 'i') { prompt         = argv[i + 1]; }
    else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
    else if (argv[i][1] == 'm') { mode           = argv[i + 1]; }
    else if (argv[i][1] == 'y') { system_prompt  = argv[i + 1]; }
    else if (argv[i][1] == 'q') { quant_path     = argv[i + 1]; }
    else if (argv[i][1] == 'c') { cb_path        = argv[i + 1]; }
    else { error_usage(); }
  }

  if (rng_seed <= 0)         rng_seed = (unsigned int)time(NULL);
  if (temperature < 0.0)     temperature = 0.0;
  if (topp < 0.0 || 1.0 < topp) topp = 0.9;
  if (steps < 0)             steps = 0;

  Transformer transformer;
  build_transformer(&transformer, checkpoint_path);
  if (steps == 0 || steps > transformer.config.seq_len)
    steps = transformer.config.seq_len;

  preload_transformer_weights(&transformer, quant_path, cb_path);

  Tokenizer tokenizer;
  build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

  Sampler sampler;
  build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

  if (strcmp(mode, "generate") == 0) {
    generate(&transformer, &tokenizer, &sampler, prompt, steps);
  } else if (strcmp(mode, "chat") == 0) {
    chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
  } else {
    fprintf(stderr, "unknown mode: %s\n", mode);
    error_usage();
  }

  free_sampler(&sampler);
  free_tokenizer(&tokenizer);
  free_transformer(&transformer);
  return 0;
}
#endif