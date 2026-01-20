# exp/proc_1（exp2 trace 映射/对外导出 v1）

本目录提供把 `exp/exp2/run_exp.py --save_hop_traces` 产出的 trace 结果，整理成“给合作者使用”的精简样本级 `.npz` 的工具（v1）。

与 `exp/proc/` 的区别：
- 去掉 `tok`（逐 token 文本片段）。
- 新增 `length`（三段 token 长度）：`[in, cot, out]`，并保证与 `span_in/span_cot/span_out` 对齐。
- `hop` 字段采用“默认策略”：当 trace 样本中存在 `vh` 时才输出 `hop`；否则不输出且不报错。
- 支持一次性处理 `exp/exp2/output/traces/` 下所有 run 目录（所有数据集-方法组合）。

---

## 输入结构（exp2 traces）

`exp2` 的 trace run 目录形如：
- `exp/exp2/output/traces/<dataset>/<model>/<run_tag>/`

每个 run 目录包含：
- `manifest.jsonl`（每行一个样本记录，包含 `file=ex_*.npz`）
- `ex_*.npz`（每样本一个 npz）

---

## 输出位置与命名

默认输出到：
- `exp/proc_1/output/<trace_dir 在 traces/ 之后的同构路径>/`

例如输入：
- `.../output/traces/exp/exp2/data/math.jsonl/qwen-8B/<run_tag>/`

默认输出：
- `exp/proc_1/output/exp/exp2/data/math.jsonl/qwen-8B/<run_tag>/`

---

## 输出 `.npz` 字段

每个输出样本 `.npz` 仅包含下列键：
- `attr`：`float32[L]`，row 归因向量；覆盖 `input+cot+output` 的有效 token（移除 generation 末尾 EOS）。
- `hop`：`float32[H, L]`（可选），当 trace npz 中存在 `vh` 时输出（同样移除 EOS，并与 `attr` 等长对齐）。
- `span_in`：`int64[2]`，input 在向量中的闭区间范围。
- `span_cot`：`int64[2]`，cot 在向量中的闭区间范围（无 cot 时为 `[-1, -1]`）。
- `span_out`：`int64[2]`，output 在向量中的闭区间范围。
- `length`：`int64[3]`，顺序为 `[in, cot, out]`，长度与 `span_*` 严格对应（闭区间长度 `end-start+1`，空 span 长度为 0）。
- `rise`：`float64`，row 的 RISE（faithfulness）。
- `mas`：`float64`，row 的 MAS（faithfulness）。
- `recovery`：`float64`，row 的 Recovery@10%（没有 recovery 时为 NaN）。

---

## 用法示例

处理 traces 下所有 run（推荐）：
```bash
python exp/proc_1/map_exp2_traces_to_proc_1.py \
  --traces_root exp/exp2/output/traces
```

只处理某一个 run 目录：
```bash
python exp/proc_1/map_exp2_traces_to_proc_1.py \
  --trace_dir exp/exp2/output/traces/exp/exp2/data/math.jsonl/qwen-8B/ifr_multi_hop_both_n1_mfaithfulness_gen_100ex
```

调试：每个 run 只处理前 5 条、允许覆盖输出：
```bash
python exp/proc_1/map_exp2_traces_to_proc_1.py \
  --traces_root exp/exp2/output/traces \
  --limit 5 \
  --overwrite
```
