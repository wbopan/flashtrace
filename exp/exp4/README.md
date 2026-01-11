# FlashTrace 实验 4（Aider 归因忠实度 / row-only）

本目录提供 Aider 数据集上的 token-level 归因忠实度评测工具，**只输出 row 部分的 RISE/MAS**，不保存样本级 trace。

评测范围（固定）：
- 数据集：`exp/exp4/data/aider.jsonl`
- 方法：
  - `ifr_all_positions`
  - `ifr_multi_hop_both`（FlashTrace）
- 指标：`RISE`、`MAS`（row attribution only）

主要文件：
- `run_exp.py`：归因 + 忠实度评测，输出到 `exp/exp4/output/`

---

## 数据格式

`exp/exp4/data/aider.jsonl` 每行一个 JSON，对应一个样本：
- `input`：prompt（直接作为 user prompt 内容）
- `output`：target（直接作为模型生成文本；脚本会内部追加 EOS 做打分）
- `length`：数据自带字段（脚本不依赖，仅透传到 metadata）

说明：Aider 的 `output` 形如：
1) 第一行 `xxx.py`
2) 第二行 opening fence ``` 
3) 中间为代码
4) 最后一行为 closing fence ``` 

---

## 归因与 sink 选择

脚本对每个样本都将 `input` 作为 `prompt`，将 `output` 作为 `target`（不做重新生成），并在归因结果上选择不同的 sink（`indices_to_explain=[start_tok,end_tok]`，均基于 `tokenizer(target, add_special_tokens=False)` 的 token span；不含 EOS）。

### `ifr_all_positions`（输出两个 sink）

- `last_line`：取 `output` 中 **closing fence 之前最后一个“非空且非 ```”行**，并将该行的字符 span 映射到 token span；若无法解析则回退为 `full_output`。
- `last_token`：取 `last_line` 的最后一个 token（单点 span `[end,end]`）。

注意：脚本会对同一个样本只计算一次 `ifr_all_positions` 的归因矩阵，然后分别在两个 sink 上取 row attribution 并计算忠实度。

### `ifr_multi_hop_both`（FlashTrace，只输出一个 sink）

- `full_output`：用完整 `output` 作为 sink（token span `[0, n_tok-1]`）。
- 忠实度扰动侧会沿用 exp2 的协议：对 prompt-side 会跳过 stop tokens（由 `ft_ifr_improve.py` 的 stop-token 配置决定）。

---

## 指标输出（row-only）

输出 CSV 仅包含 row attribution 的 `RISE/MAS` 聚合统计：
- `Method,Sink,Row_RISE_Mean,Row_RISE_Std,Row_MAS_Mean,Row_MAS_Std,Used,Skipped,Avg_Sample_Time_s`

输出路径：
- `exp/exp4/output/faithfulness/aider/<model_tag>/row_only_<N>_examples.csv`

其中 `<model_tag>` 优先取 `--model`，否则取 `--model_path` 的目录名。

---

## 使用说明

推荐从 repo root 运行（保证相对路径可用）：

```bash
python exp/exp4/run_exp.py \
  --data_path exp/exp4/data/aider.jsonl \
  --output_root exp/exp4/output \
  --model qwen-8B \
  --model_path /opt/share/models/Qwen/Qwen3-8B/ \
  --cuda 2,3,4,5,6,7 \
  --num_examples 100 \
  --n_hops 1 \
  --k 20
```

常用参数：
- `--model_path` / `--model`：本地模型路径或 HF repo id（至少提供其一）
- `--tokenizer_path`：可选；不提供则默认复用模型路径/id
- `--cuda`：支持 `0`（单卡）或 `0,1,2`（多卡，内部会设置 `CUDA_VISIBLE_DEVICES` 并用 `device_map=auto`）
- `--num_examples`：评测前 N 条（按文件顺序；`--seed` 预留，当前不做随机抽样）
- `--n_hops`：FlashTrace（`ifr_multi_hop_both`）的 hop 数
- `--k`：MAS/RISE 的扰动步数
- `--chunk_tokens` / `--sink_chunk_tokens`：IFR 计算的 chunk 参数（一般保持默认）
