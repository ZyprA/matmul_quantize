# llama2.cのFPGAカーネル実装

[llama2.c](https://github.com/karpathy/llama2.c) の推論処理をFPGAにオフロードし、高速化を行うための実装です。

## 開発環境・ターゲット

- **Target Board**: Kria KV260
- **Development Tool**: Vitis Unified IDE 2025.2
- **HLS Tool**: Vitis HLS
- **Host Program**: [karpathy/llama2.c](https://github.com/karpathy/llama2.c)

## パフォーマンス

FPGAカーネルの実装により、CPU実行時と比較して約4.5倍のスループット向上を達成しています。

| 実行環境 | 推論速度 |
| :--- | :--- |
| **CPU (Baseline)** | ~10 toks/s |
| **FPGA Kernel** | ~45 toks/s |
