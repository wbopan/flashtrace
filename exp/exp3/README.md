# FlashTrace 实验 3：长/短 CoT 对比（case study）

本目录提供一个「长/短 CoT」的最小可复现实验：
- 从 RULER `niah_mq_q2 (1024)` 中分别筛出：
  - short-CoT：短推理 + `\box{}` 最终答案
  - long-CoT：长推理 + `\box{}` 最终答案
- 只跑 `attnlrp`（hop0）并只计算 token-level `recovery@10%`（gold 来自 `needle_spans`）。
- 落盘 trace（npz + manifest）到 `exp/exp3/output/`，格式对齐 `exp/exp2/run_exp.py` 的 trace 习惯。

## 1) 采样与过滤（生成 + judge）

默认读取：
`data/ruler_multihop/1024/niah_mq_q2/validation.jsonl`

需要一个 OpenAI-compatible 的 chat API（默认 `http://localhost:4000/v1`）以及 API key。

```bash
export FLASHTRACE_API_KEY=...  # 或 OPENAI_API_KEY

python exp/exp3/sample_and_filter.py \
  --tokenizer_model /opt/share/models/Qwen/Qwen3-8B/ \
  --min_long_thinking_tokens 512 \
  --max_short_thinking_tokens 256
```

输出（默认）：
- `exp/exp3/data/niah_mq_q2_short_cot.jsonl`
- `exp/exp3/data/niah_mq_q2_long_cot.jsonl`

说明：
- 默认各采 1 条；可用 `--max_short` / `--max_long` 分别指定数量（`--max_pairs` 是两者的兼容别名）。

## 2) 归因与 recovery（AttnLRP hop0）

```bash
python exp/exp3/run_exp.py \
  --model qwen-8B \
  --model_path /opt/share/models/Qwen/Qwen3-8B/ \
  --cuda 3,4,5,7
```

输出：
- recovery CSV：`exp/exp3/output/recovery/<dataset>/<model>/attnlrp_1_examples.csv`
- trace：`exp/exp3/output/traces/<dataset>/<model>/<run_tag>/ex_*.npz` + `manifest.jsonl`
- 汇总 JSON：`exp/exp3/output/recovery/summary_<model>.json`

常用参数：
- `--top_fraction`：recovery 的 top fraction（默认 0.1）
- `--attnlrp_neg_handling drop|abs`
- `--attnlrp_norm_mode norm|no_norm`
