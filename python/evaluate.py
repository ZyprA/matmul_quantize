import argparse
import math
import os
import struct

import numpy as np


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
    dim = cfg["dim"]
    hidden_dim = cfg["hidden_dim"]
    kv_dim = cfg["kv_dim"]
    return {
        "wq": dim * dim,
        "wk": dim * kv_dim,
        "wv": dim * kv_dim,
        "wo": dim * dim,
        "w1": dim * hidden_dim,
        "w2": hidden_dim * dim,
        "w3": dim * hidden_dim,
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

        sizes = {
            "wq": dim * dim,
            "wk": dim * kv_dim,
            "wv": dim * kv_dim,
            "wo": dim * dim,
            "w1": dim * hidden_dim,
            "w2": hidden_dim * dim,
            "w3": dim * hidden_dim,
        }
        vector_counts = {name: math.ceil(sz / vector_dim) for name, sz in sizes.items()}

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

        if cb_n_clusters != n_clusters:
            raise ValueError(f"クラスタ数が一致しません: quant={n_clusters}, cb={cb_n_clusters}")
        if cb_vector_dim != vector_dim:
            raise ValueError(f"vector_dim が一致しません: quant={vector_dim}, cb={cb_vector_dim}")
        if cb_cfg["dim"] != cfg["dim"] or cb_cfg["hidden_dim"] != cfg["hidden_dim"] or cb_cfg["n_layers"] != cfg["n_layers"]:
            raise ValueError("ヘッダ情報が一致しません")

        codebooks = {name: [] for name in sizes}
        for name in sizes:
            for _ in range(n_layers):
                codebooks[name].append(np.fromfile(f, dtype=np.float32, count=n_clusters * vector_dim).reshape(n_clusters, vector_dim))

    dequant = {}
    for name, sz in sizes.items():
        layers = []
        vec_count = vector_counts[name]
        for layer in range(n_layers):
            idx = raw_idx[name][layer * vec_count : (layer + 1) * vec_count]
            layer_vectors = codebooks[name][layer][idx.astype(np.int32)]
            flat = layer_vectors.reshape(-1)[:sz]
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

    print(f"{'=' * 72}")
    print(f"  ベクトル量子化評価   クラスタ数={n_clusters} ({bits}bit)  ベクトル長={vector_dim}  レイヤー数={n_layers}")
    print(f"{'=' * 72}")
    print(f"{'Layer':<6} {'Name':<5} {'MSE':>12} {'RMSE':>10} {'PSNR(dB)':>10} {'Cosine':>8}")
    print(f"{'-' * 72}")

    for layer in range(n_layers):
        for name, sz in sizes.items():
            o = orig[name][layer * sz : (layer + 1) * sz]
            dq = dequant[name][layer * sz : (layer + 1) * sz]
            mse, rmse, psnr, cos = metrics(o, dq)
            total_stats[name].append((mse, rmse, psnr, cos))
            print(f"{layer:<6} {name.upper():<5} {mse:>12.8f} {rmse:>10.6f} {psnr:>10.2f} {cos:>8.6f}")
        print()

    print(f"{'=' * 72}")
    print(f"  重み行列別の平均")
    print(f"{'-' * 72}")
    print(f"{'Name':<5} {'MSE':>12} {'RMSE':>10} {'PSNR(dB)':>10} {'Cosine':>8}")
    print(f"{'-' * 72}")

    for name, stats in total_stats.items():
        m = np.array(stats).mean(axis=0)
        print(f"{name.upper():<5} {m[0]:>12.8f} {m[1]:>10.6f} {m[2]:>10.2f} {m[3]:>8.6f}")

    all_stats = np.array([s for stats in total_stats.values() for s in stats])
    m = all_stats.mean(axis=0)
    print(f"\n{'総合平均':<5} {m[0]:>12.8f} {m[1]:>10.6f} {m[2]:>10.2f} {m[3]:>8.6f}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ベクトル量子化精度の評価")
    parser.add_argument("bin_path", help="元モデルバイナリ (例: stories15M.bin)")
    parser.add_argument("bits", type=int, help="量子化ビット数")
    parser.add_argument("--vector-dim", type=int, default=8, help="量子化ベクトル長")
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