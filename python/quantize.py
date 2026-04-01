import struct
import numpy as np
import time
import math
import os
import argparse
from scipy.stats import norm
from sklearn.cluster import KMeans, MiniBatchKMeans

USE_MINIBATCH = False


def _idx_dtype(n_clusters: int):
    if n_clusters <= 256:
        return np.uint8
    elif n_clusters <= 65536:
        return np.uint16
    raise ValueError("n_clusters > 65536 は未サポート")


def _assign(flat: np.ndarray, codebook: np.ndarray, idx_dtype) -> np.ndarray:
    diffs = np.abs(flat[:, None] - codebook[None, :])
    return diffs.argmin(axis=1).astype(idx_dtype)


def quantize_matrix_kmeans(matrix: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    X = matrix.reshape(-1, 1)
    Cls = MiniBatchKMeans if USE_MINIBATCH else KMeans
    n_init = 1 if USE_MINIBATCH else 10
    km = Cls(n_clusters=n_clusters, random_state=42, n_init=n_init, max_iter=300)
    labels = km.fit_predict(X).astype(np.int32)
    dtype = _idx_dtype(n_clusters)
    indices  = labels.reshape(matrix.shape).astype(dtype)
    codebook = km.cluster_centers_.flatten().astype(np.float32)
    return indices, codebook



def run(bin_path: str, bits: int):
    n_clusters = 1 << bits
    idx_dtype  = np.uint8 if n_clusters <= 256 else np.uint16

    stem     = os.path.splitext(os.path.basename(bin_path))[0]
    out_dir  = f"{stem}_{bits}bit"
    os.makedirs(out_dir, exist_ok=True)
    out_quant = os.path.join(out_dir, f"{stem}_{bits}bit_quant.bin")
    out_cb    = os.path.join(out_dir, f"{stem}_{bits}bit_codebook.bin")

    print(f"入力: {bin_path}")
    print(f"クラスタ数: {n_clusters} ({bits} bit)")
    print(f"出力ディレクトリ: {out_dir}")

    with open(bin_path, "rb") as f:
        header_bytes = f.read(28)
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = \
            struct.unpack("7i", header_bytes)

        shared_weights = vocab_size > 0
        vocab_size = abs(vocab_size)
        head_size  = dim // n_heads
        kv_dim     = n_kv_heads * head_size

        def read_tensor(size):
            return np.fromfile(f, dtype=np.float32, count=size)

        token_embedding = read_tensor(vocab_size * dim)
        rms_att_weight  = read_tensor(n_layers * dim)
        wq = read_tensor(n_layers * dim * dim)
        wk = read_tensor(n_layers * dim * kv_dim)
        wv = read_tensor(n_layers * dim * kv_dim)
        wo = read_tensor(n_layers * dim * dim)
        rms_ffn_weight  = read_tensor(n_layers * dim)
        w1 = read_tensor(n_layers * dim * hidden_dim)
        w2 = read_tensor(n_layers * hidden_dim * dim)
        w3 = read_tensor(n_layers * dim * hidden_dim)

    sizes = {
        "wq": dim * dim,
        "wk": dim * kv_dim,
        "wv": dim * kv_dim,
        "wo": dim * dim,
        "w1": dim * hidden_dim,
        "w2": hidden_dim * dim,
        "w3": dim * hidden_dim,
    }
    tensors = {"wq": wq, "wk": wk, "wv": wv, "wo": wo,
               "w1": w1, "w2": w2, "w3": w3}

    q_indices  = {name: [] for name in sizes}
    q_codebook = {name: [] for name in sizes}

    t0 = time.time()
    for layer in range(n_layers):
        for name, sz in sizes.items():
            mat = tensors[name][layer * sz : (layer + 1) * sz]
            idx, cb = quantize_matrix_kmeans(mat, n_clusters)
            q_indices[name].append(idx)
            q_codebook[name].append(cb)
        print(f"  Layer {layer + 1}/{n_layers} 完了  ({time.time() - t0:.1f}s)")

    vs_out = vocab_size if shared_weights else -vocab_size

    with open(out_quant, "wb") as f:
        f.write(struct.pack("7i", dim, hidden_dim, n_layers, n_heads,
                            n_kv_heads, vs_out, seq_len))
        f.write(struct.pack("i", n_clusters))
        token_embedding.tofile(f)
        rms_att_weight.tofile(f)
        for layer in range(n_layers):
            q_indices["wq"][layer].tofile(f)
        for layer in range(n_layers):
            q_indices["wk"][layer].tofile(f)
        for layer in range(n_layers):
            q_indices["wv"][layer].tofile(f)
        for layer in range(n_layers):
            q_indices["wo"][layer].tofile(f)
        rms_ffn_weight.tofile(f)
        for layer in range(n_layers):
            q_indices["w1"][layer].tofile(f)
        for layer in range(n_layers):
            q_indices["w2"][layer].tofile(f)
        for layer in range(n_layers):
            q_indices["w3"][layer].tofile(f)

    print(f"量子化 bin を書き出し: {out_quant}")

    with open(out_cb, "wb") as f:
        f.write(struct.pack("7i", dim, hidden_dim, n_layers, n_heads,
                            n_kv_heads, vs_out, seq_len))
        f.write(struct.pack("i", n_clusters))
        for name in sizes:
            for layer in range(n_layers):
                q_codebook[name][layer].tofile(f)

    print(f"コードブック bin を書き出し: {out_cb}")
    print(f"総処理時間: {time.time() - t0:.2f} 秒")

    orig_quant_bytes = sum(n_layers * sz * 4 for sz in sizes.values())
    new_quant_bytes  = sum(n_layers * sz * (1 if n_clusters <= 256 else 2)
                           for sz in sizes.values())
    cb_bytes = len(sizes) * n_layers * n_clusters * 4
    print(f"\n--- 圧縮率 ---")
    print(f"元の重み行列サイズ  : {orig_quant_bytes / 1e6:.2f} MB")
    print(f"量子化後インデックス: {new_quant_bytes / 1e6:.2f} MB")
    print(f"コードブック        : {cb_bytes / 1e6:.2f} MB")
    print(f"合計削減率          : {(orig_quant_bytes - new_quant_bytes - cb_bytes) / orig_quant_bytes * 100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llama2.c 重みのスカラー量子化")
    parser.add_argument("bin_path", help="モデルバイナリ (例: stories15M.bin)")
    parser.add_argument("bits", type=int, help="量子化ビット数")
    args = parser.parse_args()
    run(args.bin_path, args.bits)