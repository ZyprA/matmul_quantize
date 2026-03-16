#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include "matmul_quantize_kernel.h"

// ─────────────────────────────────────────────────────────
// スカラー量子化ユーティリティ
// ─────────────────────────────────────────────────────────
static void scalar_quantize(
    const std::vector<float>& src,
    const std::vector<float>& codebook,
    std::vector<int>&         indices,
    std::vector<float>&       dequantized
) {
    const int K = static_cast<int>(codebook.size());
    indices.resize(src.size());
    dequantized.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        int best = 0;
        float best_d = std::abs(src[i] - codebook[0]);
        for (int k = 1; k < K; ++k) {
            float d = std::abs(src[i] - codebook[k]);
            if (d < best_d) { best_d = d; best = k; }
        }
        indices[i]     = best;
        dequantized[i] = codebook[best];
    }
}

// ─────────────────────────────────────────────────────────
// 精度レポート出力
// ─────────────────────────────────────────────────────────
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
    const int n = TEST_N;  // 320
    const int d = TEST_D;  // 640

    std::srand(42);

    // ── 1. 重み行列・入力ベクトル生成 ─────────────────────
    std::vector<float> W_float(d * n);
    std::vector<float> x_raw(n);
    for (float& v : W_float) v = (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 2.0f;
    for (float& v : x_raw)   v = (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 4.0f;

    // ── 2. float 参照 matmul ───────────────────────────────
    std::vector<float> y_ref(d, 0.0f);
    for (int r = 0; r < d; ++r) {
        float sum = 0.0f;
        for (int c = 0; c < n; ++c) sum += W_float[r * n + c] * x_raw[c];
        y_ref[r] = sum;
    }

    // ── 3. スカラー量子化 ──────────────────────────────────
    float w_min = *std::min_element(W_float.begin(), W_float.end());
    float w_max = *std::max_element(W_float.begin(), W_float.end());

    std::vector<float> codebook(GROUP_SIZE);
    for (int k = 0; k < GROUP_SIZE; ++k)
        codebook[k] = w_min + (w_max - w_min) * (k + 0.5f) / GROUP_SIZE;

    std::vector<int>   W_idx;
    std::vector<float> W_dequant;
    scalar_quantize(W_float, codebook, W_idx, W_dequant);

    // ── 4. 量子化後の参照 matmul ──────────────────────────
    std::vector<float> y_quant_ref(d, 0.0f);
    for (int r = 0; r < d; ++r) {
        float sum = 0.0f;
        for (int c = 0; c < n; ++c) sum += W_dequant[r * n + c] * x_raw[c];
        y_quant_ref[r] = sum;
    }

    // ── 5. カーネル入力バッファ構築 ───────────────────────
    const int row_blocks = n / ELEMENTS_BLOCK_W;      // 320/64 = 5
    const int blocks_w   = n * d / ELEMENTS_BLOCK_W;  // 320*640/64 = 3200
    const int blocks_x   = n / ELEMENTS_BLOCK_X;      // 320/16 = 20
    const int blocks_y   = d / ELEMENTS_BLOCK_Y;      // 640/16 = 40

    // ── W: 4bit インデックスを 256bit にパック ─────────────
    //   行 r のブロック j → w_packed[r * row_blocks + j]
    //   各ブロック内で要素 e を bits[e*4+3 : e*4] に格納（64要素×4bit=256bit）
    std::vector<BLOCK_W_PACKED> w_packed(blocks_w);
    for (int r = 0; r < d; ++r) {
        for (int j = 0; j < row_blocks; ++j) {
            BLOCK_W_PACKED packed = 0;
            for (int e = 0; e < ELEMENTS_BLOCK_W; ++e) {
                int idx = W_idx[r * n + j * ELEMENTS_BLOCK_W + e];
                // e 番目の4bitフィールドに書き込む（カーネルのアンパックと対称）
                packed.range(e * GROUP_BITS + GROUP_BITS - 1, e * GROUP_BITS) = idx;
            }
            w_packed[r * row_blocks + j] = packed;
        }
    }

    // ── X ─────────────────────────────────────────────────
    std::vector<BLOCK_X_IO> x_in(blocks_x);
    for (int i = 0; i < n; ++i)
        x_in[i / ELEMENTS_BLOCK_X][i % ELEMENTS_BLOCK_X] = x_raw[i];

    // ── コードブック ───────────────────────────────────────
    std::vector<W_DEQUANTIZED_TYPE> cb(GROUP_SIZE);
    for (int k = 0; k < GROUP_SIZE; ++k) cb[k] = codebook[k];

    // ── 出力バッファ ───────────────────────────────────────
    std::vector<BLOCK_Y_IO> y_out(blocks_y);

    // ── 6. カーネル呼び出し ───────────────────────────────
    matmul_quantize_kernel(
        w_packed.data(), cb.data(), x_in.data(), y_out.data(), n, d);

    // ── 7. 結果の収集 ─────────────────────────────────────
    std::vector<float> y_hw(d);
    for (int r = 0; r < d; ++r)
        y_hw[r] = static_cast<float>(y_out[r / ELEMENTS_BLOCK_Y][r % ELEMENTS_BLOCK_Y]);

    // ── 8. 精度評価 ───────────────────────────────────────
    std::cout << "=== 精度評価 (n=" << n << ", d=" << d
              << ", GROUP_SIZE=" << GROUP_SIZE << ") ===\n\n";
    print_accuracy("HW vs 量子化参照 (主評価)",       y_hw,       y_quant_ref);
    std::cout << "\n";
    print_accuracy("HW vs float 参照 (全体精度)",     y_hw,       y_ref);
    std::cout << "\n";
    print_accuracy("量子化参照 vs float 参照 (量子化誤差)", y_quant_ref, y_ref);
    std::cout << "\n";

    // ── 9. 先頭 10 要素のデバッグ出力 ────────────────────
    std::cout << "=== 先頭 10 要素の比較 ===\n";
    std::cout << "  idx  |  hw_out  | quant_ref | float_ref\n";
    std::cout << "-------|----------|-----------|-----------\n";
    for (int r = 0; r < 10 && r < d; ++r)
        printf("  %4d | %8.4f | %9.4f | %9.4f\n",
               r, y_hw[r], y_quant_ref[r], y_ref[r]);
    std::cout << "\n";

    // ── 10. PASS/FAIL 判定 ────────────────────────────────
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