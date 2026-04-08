#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include "matmul_quantize_kernel.h"

constexpr int NUM_CODEBOOKS = ELEMENTS_BLOCK_W / VECTOR_DIM * W_PORTS;  // 64
constexpr int INDICES_PER_BLOCK = ELEMENTS_BLOCK_W / VECTOR_DIM;        // 16

// 行rの列cのベクトルが使用するコードブックのインデックスを返す
static int get_codebook_index(int r, int c, int n) {
    int port = r % W_PORTS;
    int block_local_pos = (c / VECTOR_DIM) % INDICES_PER_BLOCK;
    return port * INDICES_PER_BLOCK + block_local_pos;
}

// 複数コードブック対応のベクトル量子化
static void vector_quantize_multi_cb(
    const std::vector<float>& W_float,
    const std::vector<std::vector<std::array<float, VECTOR_DIM>>>& codebooks,
    int d, int n,
    std::vector<int>&         indices,
    std::vector<float>&       dequantized
) {
    const int K = GROUP_SIZE;
    const int num_vectors = d * n / VECTOR_DIM;
    indices.resize(num_vectors);
    dequantized.resize(d * n);

    for (int r = 0; r < d; ++r) {
        for (int c = 0; c < n; c += VECTOR_DIM) {
            int vec_idx = r * (n / VECTOR_DIM) + c / VECTOR_DIM;
            int cb_idx = get_codebook_index(r, c, n);
            const auto& codebook = codebooks[cb_idx];

            int best = 0;
            float best_d = 0.0f;
            for (int v = 0; v < VECTOR_DIM; ++v) {
                float diff = W_float[r * n + c + v] - codebook[0][v];
                best_d += diff * diff;
            }

            for (int k = 1; k < K; ++k) {
                float dist = 0.0f;
                for (int v = 0; v < VECTOR_DIM; ++v) {
                    float diff = W_float[r * n + c + v] - codebook[k][v];
                    dist += diff * diff;
                }
                if (dist < best_d) { best_d = dist; best = k; }
            }

            indices[vec_idx] = best;
            for (int v = 0; v < VECTOR_DIM; ++v) {
                dequantized[r * n + c + v] = codebook[best][v];
            }
        }
    }
}

static void print_accuracy(const char* tag,
                            const std::vector<float>& hw,
                            const std::vector<float>& ref) {
    double mse = 0.0, max_err = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double e = hw[i] - ref[i];
        mse    += e * e;
        max_err = std::max(max_err, std::abs(e));
    }
    mse /= ref.size();
    double rmse = std::sqrt(mse);

    double peak = 0.0;
    for (float v : ref) peak = std::max(peak, std::abs((double)v));
    double psnr = (mse > 0.0 && peak > 0.0)
                  ? 10.0 * std::log10(peak * peak / mse)
                  : std::numeric_limits<double>::infinity();

    double dot = 0.0, norm_hw = 0.0, norm_ref = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        dot      += hw[i]  * ref[i];
        norm_hw  += hw[i]  * hw[i];
        norm_ref += ref[i] * ref[i];
    }
    double cosine = dot / (std::sqrt(norm_hw) * std::sqrt(norm_ref) + 1e-12);

    std::cout << "[" << tag << "]\n"
              << "  MSE     : " << mse      << "\n"
              << "  RMSE    : " << rmse     << "\n"
              << "  MaxErr  : " << max_err  << "\n"
              << "  PSNR    : " << psnr     << " dB\n"
              << "  Cosine  : " << cosine   << "\n";
}

int main() {
    const int n = TEST_N;
    const int d = TEST_D;

    std::srand(42);

    std::vector<float> W_float(d * n);
    std::vector<float> x_raw(n);
    for (float& v : W_float) v = (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 2.0f;
    for (float& v : x_raw)   v = (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 4.0f;

    std::vector<float> y_ref(d, 0.0f);
    for (int r = 0; r < d; ++r) {
        float sum = 0.0f;
        for (int c = 0; c < n; ++c) sum += W_float[r * n + c] * x_raw[c];
        y_ref[r] = sum;
    }

    float w_min = *std::min_element(W_float.begin(), W_float.end());
    float w_max = *std::max_element(W_float.begin(), W_float.end());

    // 64個のコードブックを生成（各ポートに16個 × 4ポート）
    // 各コードブックは2次元ベクトルのGROUP_SIZE個のエントリを持つ
    const int grid_size = static_cast<int>(std::sqrt(GROUP_SIZE));
    std::vector<std::vector<std::array<float, VECTOR_DIM>>> codebooks(NUM_CODEBOOKS);
    for (int cb = 0; cb < NUM_CODEBOOKS; ++cb) {
        codebooks[cb].resize(GROUP_SIZE);
        // コードブックごとに少し異なるオフセットを加えて多様性を持たせる
        float offset = (cb % INDICES_PER_BLOCK) * 0.01f;
        for (int k = 0; k < GROUP_SIZE; ++k) {
            int idx0 = k % grid_size;
            int idx1 = k / grid_size;
            codebooks[cb][k][0] = w_min + (w_max - w_min) * (idx0 + 0.5f) / grid_size + offset;
            codebooks[cb][k][1] = w_min + (w_max - w_min) * (idx1 + 0.5f) / grid_size + offset;
        }
    }

    std::vector<int>   W_idx;
    std::vector<float> W_dequant;
    vector_quantize_multi_cb(W_float, codebooks, d, n, W_idx, W_dequant);

    std::vector<float> y_quant_ref(d, 0.0f);
    for (int r = 0; r < d; ++r) {
        float sum = 0.0f;
        for (int c = 0; c < n; ++c) sum += W_dequant[r * n + c] * x_raw[c];
        y_quant_ref[r] = sum;
    }

    const int row_blocks     = n / ELEMENTS_BLOCK_W;             // ベクトル量子化後のブロック数
    const int rows_per_port  = d / W_PORTS;                       // 640/4  = 160
    const int blocks_per_port= rows_per_port * row_blocks;        // 160*12 = 1920
    const int blocks_x       = n / ELEMENTS_BLOCK_X;              // 384/4  = 96
    const int blocks_y       = d / ELEMENTS_BLOCK_Y;              // 640/4  = 160
    const int indices_per_block = ELEMENTS_BLOCK_W / VECTOR_DIM;  // 32/2 = 16
    const int indices_per_row = n / VECTOR_DIM;                   // 行あたりのインデックス数

    std::vector<BLOCK_W_PACKED> w_packed1(blocks_per_port);
    std::vector<BLOCK_W_PACKED> w_packed2(blocks_per_port);
    std::vector<BLOCK_W_PACKED> w_packed3(blocks_per_port);
    std::vector<BLOCK_W_PACKED> w_packed4(blocks_per_port);

    std::vector<BLOCK_W_PACKED>* w_ports[W_PORTS] = {
        &w_packed1, &w_packed2, &w_packed3, &w_packed4
    };

    for (int p = 0; p < W_PORTS; ++p) {
        for (int ri = 0; ri < rows_per_port; ++ri) {
            const int r = ri * W_PORTS + p;
            for (int j = 0; j < row_blocks; ++j) {
                BLOCK_W_PACKED packed = 0;
                for (int e = 0; e < indices_per_block; ++e) {
                    int idx = W_idx[r * indices_per_row + j * indices_per_block + e];
                    packed.range(e * GROUP_BITS + GROUP_BITS - 1, e * GROUP_BITS) = idx;
                }
                (*w_ports[p])[ri * row_blocks + j] = packed;
            }
        }
    }

    std::vector<BLOCK_X_IO> x_in(blocks_x);
    for (int i = 0; i < n; ++i)
        x_in[i / ELEMENTS_BLOCK_X][i % ELEMENTS_BLOCK_X] = x_raw[i];

    // CB_IO_TYPE = hls::vector<float, VECTOR_DIM> の配列としてコードブックを準備
    // カーネルは input_cb[i * GROUP_SIZE + j] の形式で読み込む（i=0..63, j=0..255）
    std::vector<CB_IO_TYPE> cb(NUM_CODEBOOKS * GROUP_SIZE);
    for (int i = 0; i < NUM_CODEBOOKS; ++i) {
        for (int k = 0; k < GROUP_SIZE; ++k) {
            for (int v = 0; v < VECTOR_DIM; ++v) {
                cb[i * GROUP_SIZE + k][v] = codebooks[i][k][v];
            }
        }
    }

    std::vector<BLOCK_Y_IO> y_out(blocks_y);

    matmul_quantize_kernel(
        w_packed1.data(),
        w_packed2.data(),
        w_packed3.data(),
        w_packed4.data(),
        cb.data(),
        x_in.data(),
        y_out.data(),
        n, d);

    std::vector<float> y_hw(d);
    for (int r = 0; r < d; ++r)
        y_hw[r] = static_cast<float>(y_out[r / ELEMENTS_BLOCK_Y][r % ELEMENTS_BLOCK_Y]);

    std::cout << "=== 精度評価 (n=" << n << ", d=" << d
              << ", GROUP_SIZE=" << GROUP_SIZE
              << ", VECTOR_DIM=" << VECTOR_DIM << ") ===\n\n";
    print_accuracy("HW vs 量子化参照 (主評価)",           y_hw,       y_quant_ref);
    std::cout << "\n";
    print_accuracy("HW vs float 参照 (全体精度)",         y_hw,       y_ref);
    std::cout << "\n";
    print_accuracy("量子化参照 vs float 参照 (量子化誤差)", y_quant_ref, y_ref);
    std::cout << "\n";

    std::cout << "=== 先頭 10 要素の比較 ===\n";
    std::cout << "  idx  |  hw_out  | quant_ref | float_ref\n";
    std::cout << "-------|----------|-----------|-----------\n";
    for (int r = 0; r < 10 && r < d; ++r)
        printf("  %4d | %8.4f | %9.4f | %9.4f\n",
               r, y_hw[r], y_quant_ref[r], y_ref[r]);
    std::cout << "\n";

    double mse = 0.0;
    for (int r = 0; r < d; ++r) { double e = y_hw[r] - y_quant_ref[r]; mse += e*e; }
    double rmse = std::sqrt(mse / d);

    if (rmse < 1.0) {
        std::cout << "PASS (RMSE=" << rmse << ")\n";
        return 0;
    } else {
        std::cout << "FAIL (RMSE=" << rmse << ")\n";
        return 1;
    }
}
