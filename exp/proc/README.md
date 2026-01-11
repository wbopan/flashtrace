# exp/proc（exp2 trace 映射/对外导出）

本目录提供把 `exp/exp2/run_exp.py --save_hop_traces` 产出的 trace 结果，整理成“给合作者使用”的精简样本级 `.npz` 的工具。

主要文件：
- `exp/proc/map_exp2_traces_to_proc.py`：读取 exp2 的 trace run 文件夹（`manifest.jsonl` + `ex_*.npz`），输出精简格式到 `exp/proc/output/`。

---

## 输入要求

你需要提供（或可自动推断）：
- `--trace_dir`：exp2 的 trace run 文件夹，例如：
  - `exp/exp2/output/traces/exp/exp2/data/morehopqa.jsonl/qwen-8B/ifr_all_positions_mfaithfulness_gen_95ex/`
- `--dataset_jsonl`：与该 trace run 对应的 exp2 缓存数据集（必须包含 `prompt` + `target`），例如：
  - `exp/exp2/data/morehopqa.jsonl`
- `--tokenizer_model`：与 exp2 归因时一致的 tokenizer（本地路径或模型名），例如：
  - `/opt/share/models/Qwen/Qwen3-8B/`

注意：
- 本脚本会严格复刻 exp2 的 token 对齐逻辑（prompt 前导空格、generation 用 `target + eos_token` 再 decode + offset 切片），因此 tokenizer 必须与 exp2 归因一致，否则会直接报错（长度对不上）。
- 样本匹配使用 `manifest.jsonl` 中的 `prompt_sha1/target_sha1` 对齐 `--dataset_jsonl`；所以 `--dataset_jsonl` 必须是当次 trace run 使用的那份缓存。

---

## 输出位置与命名

默认输出到：
- `exp/proc/output/<trace_dir 在 traces/ 之后的同构路径>/`

例如输入：
- `.../output/traces/exp/exp2/data/morehopqa.jsonl/qwen-8B/<run_tag>/`

默认输出：
- `exp/proc/output/exp/exp2/data/morehopqa.jsonl/qwen-8B/<run_tag>/`

你也可以用 `--out_dir` 显式指定输出目录。

输出目录内每个样本一个文件：`ex_000000.npz`、`ex_000001.npz` …

---

## 输出 `.npz` 字段（精简且仅包含必要信息）

每个输出样本 `.npz` **仅包含**下列键：
- `attr`：`float32[L]`，row 归因向量；已去掉 chat template，且去掉 EOS，仅覆盖 `input+cot+output` 的有效 token。
- `hop`：`float32[H, L]`（可选，仅 FT-IFR 类方法），逐 hop 的向量；同样已去掉 EOS，并与 `attr` 等长对齐。
- `tok`：`U[L]`，与 `attr/hop` 严格对齐的 token 文本片段序列（同样不含 chat template 与 EOS）。
- `span_in`：`int64[2]`，input 在向量中的闭区间范围。
- `span_cot`：`int64[2]`，cot 在向量中的闭区间范围（无 cot 时为 `[-1, -1]`）。
- `span_out`：`int64[2]`，output 在向量中的闭区间范围。
- `rise`：`float64`，row 的 RISE（faithfulness）。
- `mas`：`float64`，row 的 MAS（faithfulness）。
- `recovery`：`float64`，row 的 Recovery@10%（没有 recovery 时为 NaN）。

---

## 用法示例

最常用（建议显式传入 dataset 与 tokenizer）：
```bash
python exp/proc/map_exp2_traces_to_proc.py \
  --trace_dir exp/exp2/output/traces/exp/exp2/data/morehopqa.jsonl/qwen-8B/ifr_all_positions_mfaithfulness_gen_95ex \
  --dataset_jsonl exp/exp2/data/morehopqa.jsonl \
  --tokenizer_model /opt/share/models/Qwen/Qwen3-8B/
```

显式指定输出目录（避免默认同构路径）：
```bash
python exp/proc/map_exp2_traces_to_proc.py \
  --trace_dir exp/exp2/output/traces/exp/exp2/data/math.jsonl/qwen-8B/ifr_multi_hop_both_n1_mfaithfulness_gen_100ex/ \
  --dataset_jsonl exp/exp2/data/math.jsonl \
  --tokenizer_model /opt/share/models/Qwen/Qwen3-8B/ \
  --out_dir exp/proc/output/math_ifr_multi_hop_both
```

调试：只处理前 5 条、允许覆盖输出文件：
```bash
python exp/proc/map_exp2_traces_to_proc.py \
  --trace_dir ... \
  --dataset_jsonl ... \
  --tokenizer_model ... \
  --limit 5 \
  --overwrite
```

---

## 常见问题

- 报错 “Prompt/Generation token length mismatch”
  - 几乎总是 tokenizer 不一致；请确认 `--tokenizer_model` 与 exp2 归因时使用的 tokenizer 完全一致（建议直接用同一个 `--model_path`）。
- 报错 “Failed to match manifest sha1 to dataset_jsonl”
  - `--dataset_jsonl` 不是当次 trace run 使用的缓存，或缓存里没有 `target`。
- FT-IFR 方法输出缺 `hop`
  - 对 `ifr_multi_hop_stop_words/ifr_multi_hop_both/ifr_multi_hop_split_hop/ifr_in_all_gen`，exp2 trace 必须包含 `vh`；若 trace 较旧请重新跑 exp2（带 `--save_hop_traces`）。
  - 如确有需要可加 `--allow_missing_ft_hops` 强行输出（不推荐）。

