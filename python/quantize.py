import argparse
import math
import os
import struct
import time

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import KMeans, MiniBatchKMeans


USE_MINIBATCH = False

# カーネルと同じ定数
GROUP_BITS = 8
GROUP_SIZE = 1 << GROUP_BITS
VECTOR_DIM = 2
W_PORTS = 4
BITWIDTH = 128
ELEMENTS_BLOCK_W = BITWIDTH // (GROUP_BITS // VECTOR_DIM)  # = 32
NUM_CODEBOOKS = (ELEMENTS_BLOCK_W // VECTOR_DIM) * W_PORTS  # = 64


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


def quantize_matrix_multi_codebook(
    flat: np.ndarray,
    d: int,
    n: int,
    n_clusters: int,
    use_minibatch: bool,
    return_vectors: bool = False
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    カーネルの構造に合わせた複数コードブックによるベクトル量子化（NumPy化版）

    カーネルは ELEMENTS_BLOCK_W/VECTOR_DIM * W_PORTS 個のコードブックを持つ:
    - 各ポート（0〜W_PORTS-1）ごとに ELEMENTS_BLOCK_W/VECTOR_DIM 個のコードブック
    - ポート p のブロック内位置 pos のベクトルは codebook[p * 16 + pos] を使用

    Args:
        flat: flatten された重み行列 (d * n 要素)
        d: 出力次元（行数）
        n: 入力次元（列数）
        n_clusters: クラスタ数（GROUP_SIZE）
        use_minibatch: MiniBatchKMeans を使うか
        return_vectors: プロット用にベクトルも返すか

    Returns:
        indices: 量子化インデックス (d * n // VECTOR_DIM,)
        codebooks: コードブック (NUM_CODEBOOKS, n_clusters, VECTOR_DIM)
        codebook_vectors: (return_vectors=True の場合) 各コードブックのベクトル
    """
    vectors_per_block = ELEMENTS_BLOCK_W // VECTOR_DIM  # = 16

    # --- 行列を (d, blocks_per_row, vectors_per_block, VECTOR_DIM) に reshape ---
    n_padded = math.ceil(n / ELEMENTS_BLOCK_W) * ELEMENTS_BLOCK_W
    blocks_per_row = n_padded // ELEMENTS_BLOCK_W

    matrix = flat.astype(np.float32)
    if n_padded > n:
        # 行方向のパディングのみ：一度 (d, n) に戻してから列方向にパディング
        matrix = np.pad(
            matrix.reshape(d, n),
            ((0, 0), (0, n_padded - n)),
            mode="constant",
        )
    # shape: (d, blocks_per_row, vectors_per_block, VECTOR_DIM)
    matrix = matrix.reshape(d, blocks_per_row, vectors_per_block, VECTOR_DIM)

    # --- ポートごとの行インデックス（W_PORTS=4 に対応） ---
    # port_row_indices[port] = そのポートに属する行番号の配列
    port_row_indices = [np.where(np.arange(d) % W_PORTS == port)[0] for port in range(W_PORTS)]

    # --- 各コードブックのベクトルを NumPy スライスで一括収集 ---
    # codebook_vectors[cb_idx]: shape (n_port_rows * blocks_per_row, VECTOR_DIM)
    codebook_vectors: list[np.ndarray] = [None] * NUM_CODEBOOKS  # type: ignore[list-item]
    for port in range(W_PORTS):
        port_rows = port_row_indices[port]
        # shape: (len(port_rows), blocks_per_row, vectors_per_block, VECTOR_DIM)
        port_matrix = matrix[port_rows]
        for pos in range(vectors_per_block):
            cb_idx = port * vectors_per_block + pos
            # shape: (len(port_rows) * blocks_per_row, VECTOR_DIM)
            codebook_vectors[cb_idx] = port_matrix[:, :, pos, :].reshape(-1, VECTOR_DIM)

    # --- 各コードブックで k-means を並列実行 ---
    cls = MiniBatchKMeans if use_minibatch else KMeans
    n_init = 1 if use_minibatch else 10

    def fit_codebook(cb_idx: int):
        vecs = codebook_vectors[cb_idx]
        if len(vecs) < n_clusters:
            raise ValueError(
                f"コードブック {cb_idx} のサンプル数が不足: {len(vecs)} < {n_clusters}"
            )
        km = cls(n_clusters=n_clusters, random_state=42, n_init=n_init, max_iter=1000)
        labels = km.fit_predict(vecs)
        return cb_idx, km.cluster_centers_, labels

    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(fit_codebook)(cb_idx) for cb_idx in range(NUM_CODEBOOKS)
    )

    codebooks = np.zeros((NUM_CODEBOOKS, n_clusters, VECTOR_DIM), dtype=np.float32)
    codebook_labels: list[np.ndarray] = [None] * NUM_CODEBOOKS  # type: ignore[list-item]
    for cb_idx, centers, labels in results:
        codebooks[cb_idx] = centers
        codebook_labels[cb_idx] = labels

    # --- インデックス配列を NumPy で一括構築 ---
    # indices_3d[row, block, pos] = そのベクトルの量子化インデックス
    idx_dtype = _idx_dtype(n_clusters)
    indices_3d = np.zeros((d, blocks_per_row, vectors_per_block), dtype=idx_dtype)

    for port in range(W_PORTS):
        port_rows = port_row_indices[port]
        n_port_rows = len(port_rows)
        for pos in range(vectors_per_block):
            cb_idx = port * vectors_per_block + pos
            # labels shape: (n_port_rows * blocks_per_row,)
            labels_2d = codebook_labels[cb_idx].reshape(n_port_rows, blocks_per_row)
            indices_3d[port_rows, :, pos] = labels_2d

    # (d, blocks_per_row, vectors_per_block) → フラット化してパディング分を除去
    indices = indices_3d.reshape(-1)[: d * n // VECTOR_DIM]

    if return_vectors:
        return indices, codebooks, codebook_vectors
    return indices, codebooks


def plot_codebooks(
    codebook_vectors: list[np.ndarray],
    codebooks: np.ndarray,
    name: str,
    layer: int,
    out_dir: str,
    max_samples: int = 1000
):
    """
    各コードブックのベクトルとセントロイドを2次元プロット

    Args:
        codebook_vectors: 各コードブックのベクトル (NUM_CODEBOOKS 個)
        codebooks: セントロイド (NUM_CODEBOOKS, n_clusters, VECTOR_DIM)
        name: テンソル名
        layer: レイヤー番号
        out_dir: 出力ディレクトリ
        max_samples: プロットする最大サンプル数（描画高速化のため）
    """
    vectors_per_block = ELEMENTS_BLOCK_W // VECTOR_DIM  # = 16

    fig, axes = plt.subplots(W_PORTS, vectors_per_block, figsize=(32, 8))
    fig.suptitle(f"{name.upper()} Layer {layer} - Codebook Visualization (64 codebooks)", fontsize=14)

    for port in range(W_PORTS):
        for pos in range(vectors_per_block):
            cb_idx = port * vectors_per_block + pos
            ax = axes[port, pos]

            vecs = codebook_vectors[cb_idx]
            centers = codebooks[cb_idx]

            # サンプル数を制限（描画高速化）
            if len(vecs) > max_samples:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(vecs), max_samples, replace=False)
                vecs_plot = vecs[idx]
            else:
                vecs_plot = vecs

            # 元のベクトルを散布図でプロット（薄い色）
            ax.scatter(vecs_plot[:, 0], vecs_plot[:, 1], s=1, alpha=0.3, c='blue')

            # セントロイドをプロット（濃い色、大きいマーカー）
            ax.scatter(centers[:, 0], centers[:, 1], s=20, alpha=0.8, c='red', marker='x')

            ax.set_title(f"cb{cb_idx}", fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{name}_layer{layer}_codebooks.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    プロット保存: {out_path}")


def run(bin_path: str, bits: int, vector_dim: int, use_minibatch: bool, plot: bool = False):
    n_clusters = 1 << bits

    if bits != GROUP_BITS:
        raise ValueError(f"bits は {GROUP_BITS} である必要があります（カーネル設定）")
    if vector_dim != VECTOR_DIM:
        raise ValueError(f"vector_dim は {VECTOR_DIM} である必要があります（カーネル設定）")

    stem = os.path.splitext(os.path.basename(bin_path))[0]
    out_dir = f"{stem}_{bits}bit_vq{vector_dim}"
    os.makedirs(out_dir, exist_ok=True)
    out_quant = os.path.join(out_dir, f"{stem}_{bits}bit_vq{vector_dim}_quant.bin")
    out_cb = os.path.join(out_dir, f"{stem}_{bits}bit_vq{vector_dim}_codebook.bin")

    print(f"入力: {bin_path}")
    print(f"クラスタ数: {n_clusters} ({bits} bit)")
    print(f"ベクトル長: {vector_dim}")
    print(f"コードブック数: {NUM_CODEBOOKS} (ELEMENTS_BLOCK_W/VECTOR_DIM * W_PORTS = {ELEMENTS_BLOCK_W}//{VECTOR_DIM} * {W_PORTS})")
    print(f"出力ディレクトリ: {out_dir}")

    cfg, tensors = load_original(bin_path)
    sizes = tensor_sizes(cfg)
    n_layers = cfg["n_layers"]

    quantized_indices = {name: [] for name in sizes}
    codebooks = {name: [] for name in sizes}

    t0 = time.time()
    for layer in range(n_layers):
        for name, info in sizes.items():
            sz = info["size"]
            d = info["d"]
            n = info["n"]
            flat = tensors[name][layer * sz : (layer + 1) * sz]

            if plot:
                indices, cbs, cb_vectors = quantize_matrix_multi_codebook(
                    flat, d, n, n_clusters, use_minibatch, return_vectors=True
                )
                plot_codebooks(cb_vectors, cbs, name, layer, out_dir)
            else:
                indices, cbs = quantize_matrix_multi_codebook(
                    flat, d, n, n_clusters, use_minibatch
                )

            quantized_indices[name].append(indices)
            codebooks[name].append(cbs)
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

    # コードブックの書き出し
    # カーネルの load_cb と同じ順序: codebook[cb_idx][cluster_idx] の形式
    # 各テンソル・各レイヤーごとに NUM_CODEBOOKS 個のコードブック
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
        f.write(struct.pack("i", NUM_CODEBOOKS))
        for name in sizes:
            for layer in range(n_layers):
                # codebooks[name][layer] shape: (NUM_CODEBOOKS, n_clusters, VECTOR_DIM)
                # カーネル形式: cache_cb[cb_idx][cluster_idx] なので
                # cb_idx=0〜63, cluster_idx=0〜255 の順で書き出し
                for cb_idx in range(NUM_CODEBOOKS):
                    for cluster_idx in range(n_clusters):
                        codebooks[name][layer][cb_idx, cluster_idx].tofile(f)

    print(f"コードブック bin を書き出し: {out_cb}")
    print(f"総処理時間: {time.time() - t0:.2f} 秒")

    orig_quant_bytes = sum(n_layers * info["size"] * 4 for info in sizes.values())
    vector_counts = {name: math.ceil(info["size"] / vector_dim) for name, info in sizes.items()}
    new_quant_bytes = sum(n_layers * count * (1 if n_clusters <= 256 else 2) for count in vector_counts.values())
    cb_bytes = len(sizes) * n_layers * NUM_CODEBOOKS * n_clusters * vector_dim * 4
    print("\n--- 圧縮率 ---")
    print(f"元の重み行列サイズ  : {orig_quant_bytes / 1e6:.2f} MB")
    print(f"量子化後インデックス: {new_quant_bytes / 1e6:.2f} MB")
    print(f"コードブック        : {cb_bytes / 1e6:.2f} MB")
    print(f"合計削減率          : {(orig_quant_bytes - new_quant_bytes - cb_bytes) / orig_quant_bytes * 100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llama2.c 重みのベクトル量子化（複数コードブック対応）")
    parser.add_argument("bin_path", help="モデルバイナリ (例: stories15M.bin)")
    parser.add_argument("bits", type=int, nargs="?", default=GROUP_BITS, help=f"量子化ビット数（デフォルト: {GROUP_BITS}）")
    parser.add_argument("--vector-dim", type=int, default=VECTOR_DIM, help=f"量子化するベクトル長（デフォルト: {VECTOR_DIM}）")
    parser.add_argument("--minibatch", action="store_true", help="MiniBatchKMeans を使う")
    parser.add_argument("--plot", action="store_true", help="コードブックを2次元プロットで可視化")
    args = parser.parse_args()

    run(args.bin_path, args.bits, args.vector_dim, args.minibatch, args.plot)