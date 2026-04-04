#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include "matmul_quantize_kernel.h"

static void vector_quantize(
    const std::vector<float>& src,
    const std::vector<std::array<float, VECTOR_DIM>>& codebook,
    std::vector<int>&         indices,
    std::vector<float>&       dequantized
) {
    const int K = static_cast<int>(codebook.size());
    const int num_vectors = src.size() / VECTOR_DIM;
    indices.resize(num_vectors);
    dequantized.resize(src.size());

    for (int i = 0; i < num_vectors; ++i) {
        int best = 0;
        float best_d = 0.0f;
        for (int v = 0; v < VECTOR_DIM; ++v) {
            float diff = src[i * VECTOR_DIM + v] - codebook[0][v];
            best_d += diff * diff;
        }

        for (int k = 1; k < K; ++k) {
            float d = 0.0f;
            for (int v = 0; v < VECTOR_DIM; ++v) {
                float diff = src[i * VECTOR_DIM + v] - codebook[k][v];
                d += diff * diff;
            }
            if (d < best_d) { best_d = d; best = k; }
        }

        indices[i] = best;
        for (int v = 0; v < VECTOR_DIM; ++v) {
            dequantized[i * VECTOR_DIM + v] = codebook[best][v];
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

    // 2次元ベクトルのコードブックを生成（各次元を独立に均等分割）
    const int grid_size = static_cast<int>(std::sqrt(GROUP_SIZE));
    std::vector<std::array<float, VECTOR_DIM>> codebook(GROUP_SIZE);
    for (int k = 0; k < GROUP_SIZE; ++k) {
        int idx0 = k % grid_size;
        int idx1 = k / grid_size;
        codebook[k][0] = w_min + (w_max - w_min) * (idx0 + 0.5f) / grid_size;
        codebook[k][1] = w_min + (w_max - w_min) * (idx1 + 0.5f) / grid_size;
    }

    std::vector<int>   W_idx;
    std::vector<float> W_dequant;
    vector_quantize(W_float, codebook, W_idx, W_dequant);

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
    std::vector<CB_IO_TYPE> cb(GROUP_SIZE);
    for (int k = 0; k < GROUP_SIZE; ++k) {
        for (int v = 0; v < VECTOR_DIM; ++v) {
            cb[k][v] = codebook[k][v];
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