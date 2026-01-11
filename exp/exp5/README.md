# FlashTrace 实验 5：跨模型（Qwen → Llama）token-span 映射

## 背景：为什么需要映射

`exp/exp2/run_exp.py` 的归因与评估是严格 **token-level** 的，并且依赖缓存数据中的 token-span 字段：

- `indices_to_explain = [start_tok, end_tok]`（generation token indices，闭区间）
- `sink_span` / `thinking_span`（同样是 generation token spans）

这些 span 在生成缓存（`exp/exp2/sample_and_filter.py`、`exp/exp2/map_math_mine_to_exp2_cache.py`）时是用某个 tokenizer 计算并写死的（通常是 `Qwen3-8B` 的 tokenizer）。

当你切换到新模型（例如 `Llama-3.1-8B-Instruct`）时，**tokenizer 不同**，`target` 的 tokenization 长度/边界会变化，导致旧的 span 在新 tokenizer 下经常越界，从而让 exp2 在归因阶段直接报错（`IndexError: end_tok out of range`）。

## 解决方案：exp5 映射脚本

`exp/exp5/map_exp2_cache_token_spans.py` 将 exp2 缓存里的旧 token-span 从旧 tokenizer（默认 `Qwen3-8B`）映射到新 tokenizer（默认 `Llama-3.1-8B-Instruct`），并输出到：

`exp/exp5/data/<同名数据集>.jsonl`

映射策略（默认）：
1) 用旧 tokenizer 对 `target` 做 `return_offsets_mapping=True`
2) 把旧的 token-span 转成 `target` 的字符区间
3) 用新 tokenizer 对同一个 `target` 做 offsets，再把字符区间映射回新的 token-span

如遇极端情况（缓存并非由预期旧 tokenizer 产生），可启用 `--allow_fallback_answer`，用 `metadata.boxed_answer`（或 `reference_answer`）在新 tokenizer 下重新定位 span 作为兜底。

---

## Step 1：把 exp2 数据集缓存映射到 exp5/data

推荐使用仓库的 venv：

```bash
.venv/bin/python exp/exp5/map_exp2_cache_token_spans.py \
  --in_jsonl exp/exp2/data/niah_mq_q2.jsonl \
  --out_dir exp/exp5/data \
  --old_tokenizer_model /opt/share/models/Qwen/Qwen3-8B \
  --new_tokenizer_model /opt/share/models/meta-llama/Llama-3.1-8B-Instruct
```

一次映射多个数据集（示例：RULER + math）：

```bash
.venv/bin/python exp/exp5/map_exp2_cache_token_spans.py \
  --in_jsonl exp/exp2/data/niah_mq_q2.jsonl exp/exp2/data/math.jsonl \
  --out_dir exp/exp5/data \
  --old_tokenizer_model /opt/share/models/Qwen/Qwen3-8B \
  --new_tokenizer_model /opt/share/models/meta-llama/Llama-3.1-8B-Instruct
```

如果输出文件已存在，加 `--overwrite`。

默认行为：若某条样本无法映射，脚本会将其 **drop** 并在输出统计中报告；如需严格一致性请加 `--strict`（遇到首个失败样本直接退出）。如怀疑原缓存并非由 `--old_tokenizer_model` 产生，可加 `--allow_fallback_answer` 启用基于 `metadata.boxed_answer` 的兜底定位。

---

## Step 2：用 exp2 直接跑 Llama 归因评测（但数据/输出都指向 exp5）

关键点：
- **数据读取**：用 `--data_root exp/exp5/data`（让 exp2 读取映射后的缓存）
- **结果输出**：用 `--output_root exp/exp5/output`（避免写入 `exp/exp2/output`）
- **不要加** `--save_hop_traces`（避免写 trace）

### RULER（可跑 recovery + faithfulness）

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python exp/exp2/run_exp.py \
  --datasets niah_mq_q2 \
  --data_root exp/exp5/data \
  --output_root exp/exp5/output \
  --attr_funcs ifr_all_positions,attnlrp,ifr_multi_hop_both \
  --model_path /opt/share/models/meta-llama/Llama-3.1-8B-Instruct \
  --cuda 0 \
  --num_examples 100 \
  --mode faithfulness_gen,recovery_ruler
```

### math（只能跑 faithfulness；recovery 会被 exp2 显式拒绝）

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python exp/exp2/run_exp.py \
  --datasets math \
  --data_root exp/exp5/data \
  --output_root exp/exp5/output \
  --attr_funcs ifr_all_positions,attnlrp,ifr_multi_hop_both \
  --model_path /opt/share/models/meta-llama/Llama-3.1-8B-Instruct \
  --cuda 0 \
  --num_examples 100 \
  --mode faithfulness_gen
```

## 关于“是否会污染 exp2 文件夹”

- **不会污染 `exp/exp2/data/`**：我们不改 exp2 的缓存，而是输出到 `exp/exp5/data/`。
- **不加 `--save_hop_traces` 不会写 trace**。
- 但注意：`exp/exp2/run_exp.py` 本身**一定会写 CSV 指标文件**到 `--output_root`（代码行为如此，exp5 不改 exp2），所以要做到“exp2 文件夹不新增文件”，请把 `--output_root` 指向 `exp/exp5/output`（或其它目录）。

```bash
python exp/exp2/run_exp.py \
  --datasets niah_mq_q2 \
  --data_root exp/exp5/data \
  --output_root exp/exp5/output \
  --attr_funcs ifr_all_positions,attnlrp,ifr_multi_hop_both \
  --model_path /opt/share/models/meta-llama/Llama-3.1-8B-Instruct \
  --cuda 2,3,4,5,6,7 \
  --num_examples 100 \
  --mode faithfulness_gen \
  --n_hops 1
&& python exp/exp2/run_exp.py \
  --datasets math \
  --data_root exp/exp5/data \
  --output_root exp/exp5/output \
  --attr_funcs ifr_all_positions,attnlrp,ifr_multi_hop_both \
  --model_path /opt/share/models/meta-llama/Llama-3.1-8B-Instruct \
  --cuda 2,3,4,5,6,7 \
  --num_examples 100 \
  --mode faithfulness_gen \
  --n_hops 1
```