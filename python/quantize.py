import argparse
import math
import os
import struct
import time

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


USE_MINIBATCH = False


def _idx_dtype(n_clusters: int):
    if n_clusters <= 256:
        return np.uint8
    if n_clusters <= 65536:
        return np.uint16
    raise ValueError("n_clusters > 65536 は未サポート")


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
        head_size = cfg["head_size"]
        seq_len = cfg["seq_len"]

        def read_tensor(size):
            return np.fromfile(f, dtype=np.float32, count=size)

        token_embedding = read_tensor(vocab_size * dim)
        rms_att_weight = read_tensor(n_layers * dim)
        wq = read_tensor(n_layers * dim * dim)
        wk = read_tensor(n_layers * dim * kv_dim)
        wv = read_tensor(n_layers * dim * kv_dim)
        wo = read_tensor(n_layers * dim * dim)
        rms_ffn_weight = read_tensor(n_layers * dim)
        w1 = read_tensor(n_layers * dim * hidden_dim)
        w2 = read_tensor(n_layers * hidden_dim * dim)
        w3 = read_tensor(n_layers * dim * hidden_dim)
        rms_final_weight = read_tensor(dim)
        # skip freq_cis_real and freq_cis_imag (RoPE precomputed values)
        f.seek(seq_len * head_size // 2 * 4, 1)  # skip freq_cis_real
        f.seek(seq_len * head_size // 2 * 4, 1)  # skip freq_cis_imag
        if cfg["shared_weights"]:
            wcls = token_embedding
        else:
            wcls = read_tensor(vocab_size * dim)

    return cfg, {
        "token_embedding": token_embedding,
        "rms_att_weight": rms_att_weight,
        "wq": wq,
        "wk": wk,
        "wv": wv,
        "wo": wo,
        "rms_ffn_weight": rms_ffn_weight,
        "w1": w1,
        "w2": w2,
        "w3": w3,
        "rms_final_weight": rms_final_weight,
        "wcls": wcls,
    }


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


def make_vectors(flat: np.ndarray, vector_dim: int) -> tuple[np.ndarray, int]:
    original_size = int(flat.size)
    n_vectors = math.ceil(original_size / vector_dim)
    padded = np.zeros(n_vectors * vector_dim, dtype=np.float32)
    padded[:original_size] = flat.astype(np.float32, copy=False)
    return padded.reshape(n_vectors, vector_dim), original_size


def quantize_vectors(vectors: np.ndarray, n_clusters: int, use_minibatch: bool):
    if vectors.shape[0] < n_clusters:
        raise ValueError(
            f"サンプル数がクラスタ数より少なすぎます: samples={vectors.shape[0]}, clusters={n_clusters}"
        )

    cls = MiniBatchKMeans if use_minibatch else KMeans
    n_init = 1 if use_minibatch else 10
    km = cls(n_clusters=n_clusters, random_state=42, n_init=n_init, max_iter=300)
    labels = km.fit_predict(vectors).astype(np.int32)
    idx_dtype = _idx_dtype(n_clusters)
    indices = labels.astype(idx_dtype)
    codebook = km.cluster_centers_.astype(np.float32)
    return indices, codebook


def quantize_matrix_vector(flat: np.ndarray, n_clusters: int, vector_dim: int, use_minibatch: bool):
    vectors, original_size = make_vectors(flat, vector_dim)
    indices, codebook = quantize_vectors(vectors, n_clusters, use_minibatch)
    return indices, codebook, original_size


def run(bin_path: str, bits: int, vector_dim: int, use_minibatch: bool):
    n_clusters = 1 << bits

    stem = os.path.splitext(os.path.basename(bin_path))[0]
    out_dir = f"{stem}_{bits}bit_vq{vector_dim}"
    os.makedirs(out_dir, exist_ok=True)
    out_quant = os.path.join(out_dir, f"{stem}_{bits}bit_vq{vector_dim}_quant.bin")
    out_cb = os.path.join(out_dir, f"{stem}_{bits}bit_vq{vector_dim}_codebook.bin")

    print(f"入力: {bin_path}")
    print(f"クラスタ数: {n_clusters} ({bits} bit)")
    print(f"ベクトル長: {vector_dim}")
    print(f"出力ディレクトリ: {out_dir}")

    cfg, tensors = load_original(bin_path)
    sizes = tensor_sizes(cfg)
    n_layers = cfg["n_layers"]

    quantized_indices = {name: [] for name in sizes}
    codebooks = {name: [] for name in sizes}

    t0 = time.time()
    for layer in range(n_layers):
        for name, sz in sizes.items():
            flat = tensors[name][layer * sz : (layer + 1) * sz]
            indices, codebook, _ = quantize_matrix_vector(flat, n_clusters, vector_dim, use_minibatch)
            quantized_indices[name].append(indices)
            codebooks[name].append(codebook)
        print(f"  Layer {layer + 1}/{n_layers} 完了  ({time.time() - t0:.1f}s)")

    vs_out = cfg["vocab_size"] if cfg["shared_weights"] else -cfg["vocab_size"]

    with open(out_quant, "wb") as f:
        f.write(
            struct.pack(
                "7i",
                cfg["dim"],
                cfg["hidden_dim"],
                cfg["n_layers"],
                cfg["n_heads"],
                cfg["n_kv_heads"],
                vs_out,
                cfg["seq_len"],
            )
        )
        f.write(struct.pack("i", n_clusters))
        f.write(struct.pack("i", vector_dim))
        tensors["token_embedding"].tofile(f)
        tensors["rms_att_weight"].tofile(f)
        for layer in range(n_layers):
            quantized_indices["wq"][layer].tofile(f)
        for layer in range(n_layers):
            quantized_indices["wk"][layer].tofile(f)
        for layer in range(n_layers):
            quantized_indices["wv"][layer].tofile(f)
        for layer in range(n_layers):
            quantized_indices["wo"][layer].tofile(f)
        tensors["rms_ffn_weight"].tofile(f)
        for layer in range(n_layers):
            quantized_indices["w1"][layer].tofile(f)
        for layer in range(n_layers):
            quantized_indices["w2"][layer].tofile(f)
        for layer in range(n_layers):
            quantized_indices["w3"][layer].tofile(f)
        tensors["rms_final_weight"].tofile(f)
        if not cfg["shared_weights"]:
            tensors["wcls"].tofile(f)

    print(f"量子化 bin を書き出し: {out_quant}")

    with open(out_cb, "wb") as f:
        f.write(
            struct.pack(
                "7i",
                cfg["dim"],
                cfg["hidden_dim"],
                cfg["n_layers"],
                cfg["n_heads"],
                cfg["n_kv_heads"],
                vs_out,
                cfg["seq_len"],
            )
        )
        f.write(struct.pack("i", n_clusters))
        f.write(struct.pack("i", vector_dim))
        for name in sizes:
            for layer in range(n_layers):
                codebooks[name][layer].tofile(f)

    print(f"コードブック bin を書き出し: {out_cb}")
    print(f"総処理時間: {time.time() - t0:.2f} 秒")

    orig_quant_bytes = sum(n_layers * sz * 4 for sz in sizes.values())
    vector_counts = {name: math.ceil(sz / vector_dim) for name, sz in sizes.items()}
    new_quant_bytes = sum(n_layers * count * (1 if n_clusters <= 256 else 2) for count in vector_counts.values())
    cb_bytes = len(sizes) * n_layers * n_clusters * vector_dim * 4
    print("\n--- 圧縮率 ---")
    print(f"元の重み行列サイズ  : {orig_quant_bytes / 1e6:.2f} MB")
    print(f"量子化後インデックス: {new_quant_bytes / 1e6:.2f} MB")
    print(f"コードブック        : {cb_bytes / 1e6:.2f} MB")
    print(f"合計削減率          : {(orig_quant_bytes - new_quant_bytes - cb_bytes) / orig_quant_bytes * 100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llama2.c 重みのベクトル量子化")
    parser.add_argument("bin_path", help="モデルバイナリ (例: stories15M.bin)")
    parser.add_argument("bits", type=int, help="量子化ビット数")
    parser.add_argument("--vector-dim", type=int, default=8, help="量子化するベクトル長")
    parser.add_argument("--minibatch", action="store_true", help="MiniBatchKMeans を使う")
    args = parser.parse_args()

    run(args.bin_path, args.bits, args.vector_dim, args.minibatch)