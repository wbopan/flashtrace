# IFR 多跳案例分析（exp/case_study）

此目录提供一个轻量的单样本 IFR 可视化流程，不改动核心评测代码。

## 功能
- 读取单个样本（默认 `exp/exp2/data/morehopqa.jsonl`，索引 0）。
- 运行 `LLMIFRAttribution.calculate_ifr_multi_hop`。
- 输出 token 级完整文本热力图（按 input/thinking/output 分段），并将每一跳的 token 权重聚合到句子。
- 输出 JSON（完整数值）和 HTML（逐跳热力图）。

## 快速开始
```bash
# 根据本地模型修改 model/model_path
python exp/case_study/run_ifr_case.py \
  --dataset exp/exp2/data/morehopqa.jsonl \
  --index 0 \
  --model qwen-8B \
  --model_path /opt/share/models/Qwen/Qwen3-8B/ \
  --cuda 0 \
  --n_hops 1
```

产物位于 `exp/case_study/out/`，文件名形如 `ifr_case_<dataset>_idx<idx>.json/html`。

## 可选参数
- `--sink_span a b` / `--thinking_span a b`：覆盖生成侧的 sink/thinking 句子 span（默认使用缓存字段）。
- `--chunk_tokens` / `--sink_chunk_tokens`：IFR 分块参数。
- `--output_dir`：修改输出目录。

## 文件说明
- `run_ifr_case.py`：命令行入口与落盘。
- `analysis.py`：聚合与清洗（token→句子、逐跳封装）。
- `viz.py`：HTML 渲染与热力图。
