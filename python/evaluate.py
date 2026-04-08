import argparse
import math
import os
import struct

import numpy as np


# カーネルと同じ定数
GROUP_BITS = 8
GROUP_SIZE = 1 << GROUP_BITS
VECTOR_DIM = 2
W_PORTS = 4
BITWIDTH = 128
ELEMENTS_BLOCK_W = BITWIDTH // (GROUP_BITS // VECTOR_DIM)  # = 32
NUM_CODEBOOKS = (ELEMENTS_BLOCK_W // VECTOR_DIM) * W_PORTS  # = 64


def read_header(f):
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = struct.unpack("7i", f.read(28))
    shared_weights = vocab_size > 0
    vocab_size = abs(vocab_size)
    head_size = dim // n_heads
    kv_dim = n_kv_heads * head_size
    return dict(
        dim=dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        vocab_size=vocab_size,
        seq_len=seq_len,
        shared_weights=shared_weights,
        head_size=head_size,
        kv_dim=kv_dim,
    )


def load_original(path: str):
    with open(path, "rb") as f:
        cfg = read_header(f)
        dim = cfg["dim"]
        hidden_dim = cfg["hidden_dim"]
        n_layers = cfg["n_layers"]
        vocab_size = cfg["vocab_size"]
        kv_dim = cfg["kv_dim"]

        def read_tensor(size):
            return np.fromfile(f, dtype=np.float32, count=size)

        read_tensor(vocab_size * dim)
        read_tensor(n_layers * dim)
        wq = read_tensor(n_layers * dim * dim)
        wk = read_tensor(n_layers * dim * kv_dim)
        wv = read_tensor(n_layers * dim * kv_dim)
        wo = read_tensor(n_layers * dim * dim)
        read_tensor(n_layers * dim)
        w1 = read_tensor(n_layers * dim * hidden_dim)
        w2 = read_tensor(n_layers * hidden_dim * dim)
        w3 = read_tensor(n_layers * dim * hidden_dim)

    return cfg, {"wq": wq, "wk": wk, "wv": wv, "wo": wo, "w1": w1, "w2": w2, "w3": w3}


def tensor_sizes(cfg):
    """各テンソルのサイズと形状 (d, n) を返す"""
    dim = cfg["dim"]
    hidden_dim = cfg["hidden_dim"]
    kv_dim = cfg["kv_dim"]
    return {
        # (output_dim, input_dim) = (d, n)
        "wq": {"size": dim * dim, "d": dim, "n": dim},
        "wk": {"size": dim * kv_dim, "d": kv_dim, "n": dim},
        "wv": {"size": dim * kv_dim, "d": kv_dim, "n": dim},
        "wo": {"size": dim * dim, "d": dim, "n": dim},
        "w1": {"size": dim * hidden_dim, "d": hidden_dim, "n": dim},
        "w2": {"size": hidden_dim * dim, "d": dim, "n": hidden_dim},
        "w3": {"size": dim * hidden_dim, "d": hidden_dim, "n": dim},
    }


def idx_dtype(n_clusters: int):
    if n_clusters <= 256:
        return np.uint8
    if n_clusters <= 65536:
        return np.uint16
    raise ValueError("n_clusters > 65536 は未サポート")


def load_quantized(path: str, cb_path: str):
    with open(path, "rb") as f:
        cfg = read_header(f)
        dim = cfg["dim"]
        hidden_dim = cfg["hidden_dim"]
        n_layers = cfg["n_layers"]
        kv_dim = cfg["kv_dim"]
        n_clusters = struct.unpack("i", f.read(4))[0]
        vector_dim = struct.unpack("i", f.read(4))[0]

        sizes = tensor_sizes(cfg)
        vector_counts = {name: math.ceil(info["size"] / vector_dim) for name, info in sizes.items()}

        np.fromfile(f, dtype=np.float32, count=cfg["vocab_size"] * dim)
        np.fromfile(f, dtype=np.float32, count=n_layers * dim)

        raw_idx = {}
        dtype = idx_dtype(n_clusters)
        for name in ["wq", "wk", "wv", "wo"]:
            raw_idx[name] = np.fromfile(f, dtype=dtype, count=n_layers * vector_counts[name])
        np.fromfile(f, dtype=np.float32, count=n_layers * dim)
        for name in ["w1", "w2", "w3"]:
            raw_idx[name] = np.fromfile(f, dtype=dtype, count=n_layers * vector_counts[name])

    with open(cb_path, "rb") as f:
        cb_cfg = read_header(f)
        cb_n_clusters = struct.unpack("i", f.read(4))[0]
        cb_vector_dim = struct.unpack("i", f.read(4))[0]
        cb_num_codebooks = struct.unpack("i", f.read(4))[0]

        if cb_n_clusters != n_clusters:
            raise ValueError(f"クラスタ数が一致しません: quant={n_clusters}, cb={cb_n_clusters}")
        if cb_vector_dim != vector_dim:
            raise ValueError(f"vector_dim が一致しません: quant={vector_dim}, cb={cb_vector_dim}")
        if cb_num_codebooks != NUM_CODEBOOKS:
            raise ValueError(f"コードブック数が一致しません: expected={NUM_CODEBOOKS}, cb={cb_num_codebooks}")
        if cb_cfg["dim"] != cfg["dim"] or cb_cfg["hidden_dim"] != cfg["hidden_dim"] or cb_cfg["n_layers"] != cfg["n_layers"]:
            raise ValueError("ヘッダ情報が一致しません")

        # コードブック読み込み: [name][layer][cb_idx] -> (n_clusters, vector_dim)
        codebooks = {name: [] for name in sizes}
        for name in sizes:
            for _ in range(n_layers):
                layer_cbs = np.zeros((NUM_CODEBOOKS, n_clusters, vector_dim), dtype=np.float32)
                for cb_idx in range(NUM_CODEBOOKS):
                    for cluster_idx in range(n_clusters):
                        layer_cbs[cb_idx, cluster_idx] = np.fromfile(f, dtype=np.float32, count=vector_dim)
                codebooks[name].append(layer_cbs)

    # 複数コードブックを使った dequantize
    vectors_per_block = ELEMENTS_BLOCK_W // vector_dim  # = 16

    dequant = {}
    for name, info in sizes.items():
        sz = info["size"]
        d = info["d"]
        n = info["n"]
        vec_count = vector_counts[name]

        n_padded = math.ceil(n / ELEMENTS_BLOCK_W) * ELEMENTS_BLOCK_W
        blocks_per_row = n_padded // ELEMENTS_BLOCK_W

        layers = []
        for layer in range(n_layers):
            idx = raw_idx[name][layer * vec_count : (layer + 1) * vec_count].astype(np.int32)
            layer_cbs = codebooks[name][layer]

            # 行列を復元
            matrix = np.zeros((d, n_padded), dtype=np.float32)
            vec_idx = 0
            for row_idx in range(d):
                port = row_idx % W_PORTS
                for block_idx in range(blocks_per_row):
                    for pos in range(vectors_per_block):
                        if vec_idx < len(idx):
                            cb_idx = port * vectors_per_block + pos
                            cluster_idx = idx[vec_idx]
                            col_start = block_idx * ELEMENTS_BLOCK_W + pos * vector_dim
                            matrix[row_idx, col_start:col_start + vector_dim] = layer_cbs[cb_idx, cluster_idx]
                        vec_idx += 1

            flat = matrix[:, :n].reshape(-1)
            layers.append(flat)
        dequant[name] = np.concatenate(layers)

    return cfg, n_clusters, vector_dim, dequant, sizes


def metrics(orig: np.ndarray, dq: np.ndarray):
    diff = orig.astype(np.float64) - dq.astype(np.float64)
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    peak = float(np.max(np.abs(orig)))
    psnr = float(10 * np.log10(peak ** 2 / mse)) if mse > 0 else float("inf")
    cos = float(np.dot(orig, dq) / (np.linalg.norm(orig) * np.linalg.norm(dq) + 1e-12))
    return mse, rmse, psnr, cos


def evaluate(orig_path: str, quant_path: str, cb_path: str):
    cfg_o, orig = load_original(orig_path)
    cfg_q, n_clusters, vector_dim, dequant, sizes = load_quantized(quant_path, cb_path)

    n_layers = cfg_o["n_layers"]
    bits = int(np.log2(n_clusters))
    total_stats = {name: [] for name in sizes}

    print(f"{'=' * 80}")
    print(f"  ベクトル量子化評価   クラスタ数={n_clusters} ({bits}bit)  ベクトル長={vector_dim}  コードブック数={NUM_CODEBOOKS}")
    print(f"{'=' * 80}")
    print(f"{'Layer':<6} {'Name':<5} {'MSE':>12} {'RMSE':>10} {'PSNR(dB)':>10} {'Cosine':>8}")
    print(f"{'-' * 80}")

    for layer in range(n_layers):
        for name, info in sizes.items():
            sz = info["size"]
            o = orig[name][layer * sz : (layer + 1) * sz]
            dq = dequant[name][layer * sz : (layer + 1) * sz]
            mse, rmse, psnr, cos = metrics(o, dq)
            total_stats[name].append((mse, rmse, psnr, cos))
            print(f"{layer:<6} {name.upper():<5} {mse:>12.8f} {rmse:>10.6f} {psnr:>10.2f} {cos:>8.6f}")
        print()

    print(f"{'=' * 80}")
    print(f"  重み行列別の平均")
    print(f"{'-' * 80}")
    print(f"{'Name':<5} {'MSE':>12} {'RMSE':>10} {'PSNR(dB)':>10} {'Cosine':>8}")
    print(f"{'-' * 80}")

    for name, stats in total_stats.items():
        m = np.array(stats).mean(axis=0)
        print(f"{name.upper():<5} {m[0]:>12.8f} {m[1]:>10.6f} {m[2]:>10.2f} {m[3]:>8.6f}")

    all_stats = np.array([s for stats in total_stats.values() for s in stats])
    m = all_stats.mean(axis=0)
    print(f"\n{'総合平均':<5} {m[0]:>12.8f} {m[1]:>10.6f} {m[2]:>10.2f} {m[3]:>8.6f}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ベクトル量子化精度の評価（複数コードブック対応）")
    parser.add_argument("bin_path", help="元モデルバイナリ (例: stories15M.bin)")
    parser.add_argument("bits", type=int, nargs="?", default=GROUP_BITS, help=f"量子化ビット数（デフォルト: {GROUP_BITS}）")
    parser.add_argument("--vector-dim", type=int, default=VECTOR_DIM, help=f"量子化ベクトル長（デフォルト: {VECTOR_DIM}）")
    parser.add_argument("--quant", default=None, help="量子化 bin の直接パス指定")
    parser.add_argument("--cb", default=None, help="コードブック bin の直接パス指定")
    args = parser.parse_args()

    stem = os.path.splitext(os.path.basename(args.bin_path))[0]
    quant_dir = f"{stem}_{args.bits}bit_vq{args.vector_dim}"
    quant_path = args.quant or os.path.join(quant_dir, f"{stem}_{args.bits}bit_vq{args.vector_dim}_quant.bin")
    cb_path = args.cb or os.path.join(quant_dir, f"{stem}_{args.bits}bit_vq{args.vector_dim}_codebook.bin")

    for p in [args.bin_path, quant_path, cb_path]:
        if not os.path.exists(p):
            print(f"[ERROR] ファイルが見つかりません: {p}")
            raise SystemExit(1)

    evaluate(args.bin_path, quant_path, cb_path)