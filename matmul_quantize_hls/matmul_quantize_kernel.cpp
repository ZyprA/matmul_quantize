#include "matmul_quantize_kernel.h"
#include "hls_stream.h"
#include <cmath>

using BLOCK_XW = hls::vector<X_INTERNAL_TYPE, ELEMENTS_BLOCK_W>;

constexpr int ceil_log2_const(int value, int bits = 0) {
    return (value <= (1 << bits)) ? bits : ceil_log2_const(value, bits + 1);
}

auto constexpr ROW_BLOCKS_BITS = ceil_log2_const(MAX_N / ELEMENTS_BLOCK_W);
auto constexpr BLOCKS_X_BITS = ceil_log2_const(MAX_N / ELEMENTS_BLOCK_X);
auto constexpr BLOCKS_Y_BITS = ceil_log2_const(MAX_D / ELEMENTS_BLOCK_Y);
auto constexpr BLOCKS_W_BITS = ceil_log2_const(MAX_N * MAX_D / (ELEMENTS_BLOCK_W*W_PORTS));
auto constexpr ROWS_BITS = ceil_log2_const(MAX_D / W_PORTS);

static void load_x(
    const BLOCK_X_IO* input_x_io_block,
    BLOCK_X_INTERNAL* cache_x,
    ap_uint<BLOCKS_X_BITS> blocks_x
) {
    for (int i = 0; i < blocks_x; i++) {
        #pragma HLS PIPELINE II=1
        BLOCK_X_IO x_io_block = input_x_io_block[i];
        BLOCK_X_INTERNAL x_internal;
        for (int j = 0; j < ELEMENTS_BLOCK_X; j++) {
            #pragma HLS UNROLL
            x_internal[j] = (X_INTERNAL_TYPE) x_io_block[j];
        }
        cache_x[i] = x_internal;
    }
}

static void broadcast_x(
    const BLOCK_X_INTERNAL* cache_x,
    hls::stream<BLOCK_XW> stream_xw[W_PORTS],
    ap_uint<ROW_BLOCKS_BITS> row_blocks,
    ap_uint<ROWS_BITS> rows
) {
    for (int i = 0; i < rows; i++) {
        for(int j = 0; j < row_blocks; j++) {
            #pragma HLS PIPELINE II=1
            BLOCK_XW xw;
            for (int k = 0; k < NUM_INST_X; k++) {
                #pragma HLS UNROLL
                BLOCK_X_INTERNAL x = cache_x[NUM_INST_X*j + k];
                for (int l = 0; l < ELEMENTS_BLOCK_X; l++) {
                    #pragma HLS UNROLL
                    xw[k*ELEMENTS_BLOCK_X + l] = x[l];
                }
            }
            for (int k = 0; k < W_PORTS; k++) {
                #pragma HLS UNROLL
                stream_xw[k] << xw;
            }
        }
    }
}

static void load_cb(
    const CB_IO_TYPE* input_cb,
    CB_INTERNAL_TYPE cache_cb[ELEMENTS_BLOCK_W/VECTOR_DIM * W_PORTS][GROUP_SIZE]
) {
    for (int i = 0; i < GROUP_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        CB_IO_TYPE cb_io = input_cb[i];
        CB_INTERNAL_TYPE cb_internal;
        for (int j = 0; j < VECTOR_DIM; j++) {
            cb_internal[j] = (W_INTERNAL_TYPE) cb_io[j];
        }
        for (int j = 0; j < ELEMENTS_BLOCK_W/VECTOR_DIM * W_PORTS; j++) {
            #pragma HLS UNROLL
            cache_cb[j][i] = cb_internal;
        }
    }
}

static void load_w_idx_unit(
    const BLOCK_W_PACKED* input_w_io_block,
    hls::stream<BLOCK_W_INDEX>& stream_w_idx,
    ap_uint<BLOCKS_W_BITS> blocks_w
) {
    for (int i = 0; i < blocks_w; i++) {
        #pragma HLS PIPELINE II=1
        BLOCK_W_PACKED packed = input_w_io_block[i];
        BLOCK_W_INDEX idx;
        for (int j = 0; j < ELEMENTS_BLOCK_W / VECTOR_DIM; j++) {
            #pragma HLS UNROLL
            idx[j] = packed.range(j * GROUP_BITS + GROUP_BITS - 1, j * GROUP_BITS);
        }
        stream_w_idx << idx;
    }
}

static void load_w_idx(
    const BLOCK_W_PACKED* input_w1_io_block,
    const BLOCK_W_PACKED* input_w2_io_block,
    const BLOCK_W_PACKED* input_w3_io_block,
    const BLOCK_W_PACKED* input_w4_io_block,
    hls::stream<BLOCK_W_INDEX> stream_w_idx[W_PORTS],
    ap_uint<BLOCKS_W_BITS> blocks_w
) {
    load_w_idx_unit(input_w1_io_block, stream_w_idx[0], blocks_w);
    load_w_idx_unit(input_w2_io_block, stream_w_idx[1], blocks_w);
    load_w_idx_unit(input_w3_io_block, stream_w_idx[2], blocks_w);
    load_w_idx_unit(input_w4_io_block, stream_w_idx[3], blocks_w);
}

template<int W_PORT>
static void dequantize_w_unit(
    hls::stream<BLOCK_W_INDEX>& stream_w_idx,
    const CB_INTERNAL_TYPE cache_cb[ELEMENTS_BLOCK_W/VECTOR_DIM * W_PORTS][GROUP_SIZE],
    hls::stream<BLOCK_W_INTERNAL>& stream_w_internal,
    ap_uint<BLOCKS_W_BITS> blocks_w
) {
    const int offset = W_PORT * ELEMENTS_BLOCK_W/VECTOR_DIM;
    for (int i = 0; i < blocks_w; i++) {
        #pragma HLS PIPELINE II=1
        BLOCK_W_INDEX idx = stream_w_idx.read();
        BLOCK_W_INTERNAL w;
        for (int j = 0; j < ELEMENTS_BLOCK_W/VECTOR_DIM; j++) {
            #pragma HLS UNROLL
            CB_INTERNAL_TYPE cb_value = cache_cb[j + offset][idx[j]];
            for (int k = 0; k < VECTOR_DIM; k++) {
                #pragma HLS UNROLL
                w[VECTOR_DIM*j+k] = cb_value[k];
            }
        }
        stream_w_internal << w;
    }
}

static void dequantize_w(
    hls::stream<BLOCK_W_INDEX> stream_w_idx[W_PORTS],
    const CB_INTERNAL_TYPE cache_cb[ELEMENTS_BLOCK_W/VECTOR_DIM * W_PORTS][GROUP_SIZE],
    hls::stream<BLOCK_W_INTERNAL> stream_w_internal[W_PORTS],
    ap_uint<BLOCKS_W_BITS> blocks_w
) {
    #pragma HLS DATAFLOW
    dequantize_w_unit<0>(stream_w_idx[0], cache_cb, stream_w_internal[0], blocks_w);
    dequantize_w_unit<1>(stream_w_idx[1], cache_cb, stream_w_internal[1], blocks_w);
    dequantize_w_unit<2>(stream_w_idx[2], cache_cb, stream_w_internal[2], blocks_w);
    dequantize_w_unit<3>(stream_w_idx[3], cache_cb, stream_w_internal[3], blocks_w);
    
}

static void calculate_wx_unit(
    hls::stream<BLOCK_W_INTERNAL>& stream_w_internal,
    hls::stream<BLOCK_XW>& stream_xw,
    hls::stream<Y_IO_TYPE>& stream_y_io,
    ap_uint<ROW_BLOCKS_BITS> row_blocks,
    ap_uint<ROWS_BITS> rows
) {
    for (int i = 0; i < rows; i++) {
        Y_INTERNAL_TYPE acc = 0;
        for (int j = 0; j < row_blocks; j++) {
            #pragma HLS PIPELINE II=1
            BLOCK_W_INTERNAL w = stream_w_internal.read();
            BLOCK_XW xw = stream_xw.read();
            for (int k = 0; k < ELEMENTS_BLOCK_W; k++) {
                #pragma HLS UNROLL
                Y_INTERNAL_TYPE mul = w[k] * xw[k];
                acc += mul;
            }
            if (j == row_blocks - 1) { // 外ループに書くとloop_flattenされずに全体でII=1を達成できずに前のstream_xがFULLになる可能性がある
                stream_y_io << (Y_IO_TYPE) acc;
            }
        }
    }
}

static void calculate_wx(
    hls::stream<BLOCK_W_INTERNAL> stream_w_internal[W_PORTS],
    hls::stream<BLOCK_XW> stream_xw[W_PORTS],
    hls::stream<Y_IO_TYPE> stream_y_io[W_PORTS],
    ap_uint<ROW_BLOCKS_BITS> row_blocks,
    ap_uint<ROWS_BITS> rows
) {
    #pragma HLS DATAFLOW
    calculate_wx_unit(stream_w_internal[0], stream_xw[0], stream_y_io[0], row_blocks, rows);
    calculate_wx_unit(stream_w_internal[1], stream_xw[1], stream_y_io[1], row_blocks, rows);
    calculate_wx_unit(stream_w_internal[2], stream_xw[2], stream_y_io[2], row_blocks, rows);
    calculate_wx_unit(stream_w_internal[3], stream_xw[3], stream_y_io[3], row_blocks, rows);
}

static void write_y(
    hls::stream<Y_IO_TYPE> stream_y_io[W_PORTS],
    BLOCK_Y_IO* output_y_io_block,
    ap_uint<BLOCKS_Y_BITS> blocks_y
) {
    for (int i = 0; i < blocks_y; i++) {
        BLOCK_Y_IO y_io_block;
        for (int j = 0; j < ELEMENTS_BLOCK_Y / W_PORTS; j++) {
            #pragma HLS PIPELINE II=1
            for (int p = 0; p < W_PORTS; p++) {
                #pragma HLS UNROLL
                y_io_block[j * W_PORTS + p] = stream_y_io[p].read();
            }
        }
        output_y_io_block[i] = y_io_block;
    }
}

extern "C" {
    void matmul_quantize_kernel(
        const BLOCK_W_PACKED*        input_w1_io_block,
        const BLOCK_W_PACKED*        input_w2_io_block,
        const BLOCK_W_PACKED*        input_w3_io_block,
        const BLOCK_W_PACKED*        input_w4_io_block,
        const CB_IO_TYPE*    input_cb,
        const BLOCK_X_IO*            input_x_io_block,
        BLOCK_Y_IO*                  output_y_io_block,
        int n,
        int d
    ) {
        #pragma HLS INTERFACE m_axi port=input_w1_io_block  bundle=gmem0 depth=AXI_W_DEPTH  max_read_burst_length=BURST_READ_W
        #pragma HLS INTERFACE m_axi port=input_w2_io_block  bundle=gmem1 depth=AXI_W_DEPTH  max_read_burst_length=BURST_READ_W
        #pragma HLS INTERFACE m_axi port=input_w3_io_block  bundle=gmem2 depth=AXI_W_DEPTH  max_read_burst_length=BURST_READ_W
        #pragma HLS INTERFACE m_axi port=input_w4_io_block  bundle=gmem3 depth=AXI_W_DEPTH  max_read_burst_length=BURST_READ_W
        #pragma HLS INTERFACE m_axi port=input_cb          bundle=gmem4 depth=AXI_CB_DEPTH max_read_burst_length=BURST_READ_CB
        #pragma HLS INTERFACE m_axi port=input_x_io_block  bundle=gmem5 depth=AXI_X_DEPTH  max_read_burst_length=BURST_READ_X
        #pragma HLS INTERFACE m_axi port=output_y_io_block bundle=gmem5 depth=AXI_Y_DEPTH  max_write_burst_length=BURST_WRITE_Y
        #pragma HLS INTERFACE s_axilite port=n      bundle=control
        #pragma HLS INTERFACE s_axilite port=d      bundle=control
        #pragma HLS INTERFACE s_axilite port=return bundle=control

        BLOCK_X_INTERNAL cache_x[MAX_N / ELEMENTS_BLOCK_X];
        #pragma HLS BIND_STORAGE variable=cache_x type=ram_1p impl=lutram
        #pragma HLS ARRAY_PARTITION variable=cache_x cyclic factor=NUM_INST_X 

        CB_INTERNAL_TYPE cache_cb[ELEMENTS_BLOCK_W/VECTOR_DIM * W_PORTS][GROUP_SIZE];
        #pragma HLS ARRAY_PARTITION variable=cache_cb dim=1 complete
        #pragma HLS BIND_STORAGE variable=cache_cb type=ram_1p impl=lutram // type=1wnrにした場合HLSがうまいことバンク複製をしてくれなかった．明示的にバンクを作るように実施


        hls::stream<BLOCK_W_INTERNAL> stream_w_internal[W_PORTS];
        #pragma HLS STREAM variable=stream_w_internal depth=4

        hls::stream<BLOCK_W_INDEX> stream_w_idx[W_PORTS];
        #pragma HLS STREAM variable=stream_w_idx depth=4

        hls::stream<BLOCK_XW> stream_xw[W_PORTS];
        #pragma HLS STREAM variable=stream_xw depth=4

        hls::stream<Y_IO_TYPE> stream_y_io[W_PORTS];
        #pragma HLS STREAM variable=stream_y_io depth=4

        #pragma HLS DATAFLOW

        ap_uint<BLOCKS_X_BITS> blocks_x = n / ELEMENTS_BLOCK_X;
        ap_uint<ROW_BLOCKS_BITS> row_blocks = n / ELEMENTS_BLOCK_W;
        ap_uint<BLOCKS_W_BITS> blocks_w = n * d / (ELEMENTS_BLOCK_W * W_PORTS);
        ap_uint<ROWS_BITS> rows = d / W_PORTS;
        ap_uint<BLOCKS_Y_BITS> blocks_y = d / ELEMENTS_BLOCK_Y;

        load_x(input_x_io_block, cache_x, blocks_x);
        load_cb(input_cb, cache_cb);
        load_w_idx(input_w1_io_block, input_w2_io_block, input_w3_io_block, input_w4_io_block, stream_w_idx, blocks_w);
        dequantize_w(stream_w_idx, cache_cb, stream_w_internal, blocks_w);
        broadcast_x(cache_x, stream_xw, row_blocks, rows);
        calculate_wx(stream_w_internal, stream_xw, stream_y_io, row_blocks, rows);
        write_y(stream_y_io, output_y_io_block, blocks_y);
    }
}