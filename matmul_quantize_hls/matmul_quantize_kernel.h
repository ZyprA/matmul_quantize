#pragma once

#include <ap_int.h>
#include <ap_float.h>
#include "hls_vector.h"
#include <cmath>

/*
    matmul_quantize_kernelはGROUP_BITSビットコードブックベース量子化のみをサポートします．
    ap_fixedによる内部計算は1サイクルを満たすためバンク別のアキュミュレータ実装を不要とします．
    内部演算の精度を上げるにはap_fixedでの整数部とビット幅の指定を調整する必要があります．
    内部演算を1サイクルで完了できない型定義で実装をする場合，calc-floatブランチのカーネルを使用する必要があります．
*/

auto constexpr GROUP_BITS = 8; 
auto constexpr GROUP_SIZE = 1 << GROUP_BITS;

using W_INTERNAL_TYPE = ap_fixed<32, 5>;
using W_QUANTIZED_TYPE = ap_uint<GROUP_BITS>;
using W_DEQUANTIZED_TYPE = float;

using X_INTERNAL_TYPE = ap_fixed<32,13>;
using Y_INTERNAL_TYPE = ap_fixed<48,18>;
using X_IO_TYPE = float;
using Y_IO_TYPE = float;

// 定数設定
auto constexpr W_PORTS = 4;
auto constexpr BITWIDTH = 128;
auto constexpr MAX_N = 1408; // 768: 15M or 1408: 15M or 42M
auto constexpr MAX_D = 32000;
auto constexpr ELEMENTS_BLOCK_W = BITWIDTH / GROUP_BITS;
auto constexpr ELEMENTS_BLOCK_X = BITWIDTH / (sizeof(X_IO_TYPE) * 8);
auto constexpr ELEMENTS_BLOCK_Y = BITWIDTH / (sizeof(Y_IO_TYPE) * 8);
auto constexpr NUM_INST_X = ELEMENTS_BLOCK_W / ELEMENTS_BLOCK_X;
auto constexpr BURST_READ_W = 2 * 1024 / (BITWIDTH / 8);
auto constexpr BURST_READ_X = 2 * 1024 / (BITWIDTH / 8);
auto constexpr BURST_READ_CB = GROUP_SIZE;
auto constexpr BURST_WRITE_Y = 2 * 1024 / (BITWIDTH / 8);

// ブロックの定義
using BLOCK_W_INTERNAL = hls::vector<W_INTERNAL_TYPE, ELEMENTS_BLOCK_W>;
using BLOCK_W_QUANTIZED = hls::vector<W_QUANTIZED_TYPE, ELEMENTS_BLOCK_W>;
using BLOCK_W_DEQUANTIZED = hls::vector<W_DEQUANTIZED_TYPE, ELEMENTS_BLOCK_W>;
using BLOCK_W_PACKED   = ap_uint<ELEMENTS_BLOCK_W * GROUP_BITS>;
using BLOCK_X_INTERNAL = hls::vector<X_INTERNAL_TYPE, ELEMENTS_BLOCK_X>;
using BLOCK_Y_INTERNAL = hls::vector<Y_INTERNAL_TYPE, ELEMENTS_BLOCK_Y>;
using BLOCK_X_IO = hls::vector<X_IO_TYPE, ELEMENTS_BLOCK_X>;
using BLOCK_Y_IO = hls::vector<Y_IO_TYPE, ELEMENTS_BLOCK_Y>;

// TBの定義
auto constexpr TEST_N = 128 * 3;
auto constexpr TEST_D = 640;

auto constexpr AXI_W_DEPTH = TEST_N * TEST_D / (ELEMENTS_BLOCK_W * W_PORTS);
auto constexpr AXI_X_DEPTH = TEST_N / ELEMENTS_BLOCK_X;
auto constexpr AXI_Y_DEPTH = TEST_D / ELEMENTS_BLOCK_Y;
auto constexpr AXI_CB_DEPTH = GROUP_SIZE;

// カーネルの定義
extern "C" {
    void matmul_quantize_kernel(
        const BLOCK_W_PACKED*        input_w1_io_block,
        const BLOCK_W_PACKED*        input_w2_io_block,
        const BLOCK_W_PACKED*        input_w3_io_block,
        const BLOCK_W_PACKED*        input_w4_io_block,
        const W_DEQUANTIZED_TYPE*    input_cb,
        const BLOCK_X_IO*            input_x_io_block,
        BLOCK_Y_IO*                  output_y_io_block,
        int n,
        int d
    );
}