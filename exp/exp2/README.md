# FlashTrace 实验 2（多步推理下的忠实度）

本目录提供「11 数据集 × 9 方法 × 3 指标」的实验工具，**跳过 AT2**，**跳过 math**。流程分为两步：先采样并过滤高质量 CoT+boxed 生成，再对过滤结果做归因评估。

支持数据集：MoreHopQA、HotpotQA（RULER hotpotqa_long）、RULER niah（niah_*）、RULER variable tracking（vt_*）。RULER 路径自动在 `data/ruler_multihop/<len>/.../validation.jsonl` 中搜索。

主要文件：
- `sample_and_filter.py`：采样 + 判定一致性，输出到 `exp/exp2/data/`
- `run_exp.py`：归因测试，输出到 `exp/exp2/output/`
- `dataset_utils.py`：数据加载、答案 span 解析

采样脚本支持的数据集
- `morehopqa`（本地 `data/with_human_verification.json`）
- `hotpotqa_long`（自动在 `data/ruler_multihop/<len>/hotpotqa_long/validation.jsonl` 搜索）
- `niah_*`（RULER niah 变体，自动搜索同上）
- `vt_*`（RULER variable tracking 变体，自动搜索同上）
- 直接传 RULER JSONL 路径（作为数据集名处理），其余类型不支持

归因测试支持
- 数据集：优先使用 `exp/exp2/data/<name>.jsonl` 缓存，若无则按采样同样的解析规则加载；math 显式拒绝。
- 指标：
  - `faithfulness_gen`（生成侧）：可运行在任何已加载样本（math 以外）。
- 方法（`--attr_funcs`）：`IG`、`perturbation_all`、`perturbation_CLP`、`perturbation_REAGENT`、`attention`（内部融合 IG）、`ifr_all_positions`、`ifr_multi_hop`、`attnlrp`、`ft_attnlrp`、`basic`。AT2 未提供。

---

## 数据采样

实现逻辑
- 统一数据加载：`DatasetLoader` 读取 MoreHopQA / HotpotQA / RULER niah / RULER vt；可直接传自定义 RULER JSONL。
- 生成模型：`qwen3-235b-a22b-2507`（英文 system prompt），要求「先简要思考，再用 `\box{}` 包裹最终答案且末尾不追加内容」；user prompt 为原题，无额外模板。
- 判定模型：`deepseek-v3-1-terminus`（英文 system prompt），只输出 True/False 判断 `\box{}` 内文与参考答案是否一致。
- 过滤：仅保留「思考 + 末尾 boxed 答案」且判定为 True 的样本；`target` 用提取的思考片段与 **去掉 box 包裹的最终答案** 重组，附带 token 级 `sink_span`/`thinking_span`、`reference_answer`、`judge_response`（不再存 `candidate_answer`），`indices_to_explain` 统一写为 `sink_span`（boxed 内文在 `target` 的 generation token span，[start_tok, end_tok]）。
- 采样会按原始顺序依次尝试样本，判定失败立即跳过；累计到 `--max_examples` 条成功样本即提前停止（若源数据不足则更少），tqdm 会分别显示尝试与成功计数。

使用说明
```bash
export FLASHTRACE_API_KEY=sk-yaojia-get-ccfa  # 或 OPENAI_API_KEY

# 示例：采样 hotpotqa_long，保留最多 100 条判定为 True 的样本
python exp/exp2/sample_and_filter.py \
  --dataset data/ruler_multihop/4096/hotpotqa_long/validation.jsonl \
  --max_examples 100 \
  --api_key sk-yaojia-get-ccfa \
  --tokenizer_model /opt/share/models/Qwen/Qwen3-8B > exp/exp2/out.log
```
常用参数：
- `--dataset`：morehopqa | hotpotqa_long | niah_* | vt_*（或直接 JSONL 路径）
- `--max_examples`：希望保留的成功样本数；达到后即停止（若源数据不足则更少）
- `--tokenizer_model`：用于 span 检测的 tokenizer（默认复用生成模型）
- `--api_base`/`--api_key`：接口地址与密钥（默认本地 http://localhost:4000/v1）
- `--request_interval` / `--judge_interval`：生成/判定间隔节流（默认 1s）
- `--rate_limit_delay`：遇到 HTTP 429 时的等待秒数（默认 5s）；会在重试前自动 sleep
输出：`exp/exp2/data/<dataset>.jsonl`

---

## 归因测试

实现逻辑
- 输入：优先读取 `exp/exp2/data/<dataset>.jsonl`（过滤缓存）；若不存在则回退到原始数据解析。
- 方法：忠实度（token-level RISE/MAS）对齐 `evaluations/faithfulness.py` 的逻辑（AT2 未实现），math 自动拒绝。
- 多跳 FlashTrace：若缓存含 `sink_span`/`thinking_span` 则用于 multi-hop IFR，否则默认整句答案为 sink。
- 批大小估算：沿用原脚本 `(max_input_len-100)/len(tokenizer(format_prompt(prompt)+target))` 的保守估计（至少 1）。`max_input_len` 由代码内置映射表基于 `--model` 字符串决定，未命中或仅传 `--model_path` 时默认 2000；如需映射值而又用本地路径，请同时传入对应的 `--model` 名称。
- 计时：对每个样本的归因计算（coverage/faithfulness）分别计时，最终在 CSV 末尾追加 `Avg Sample Time (s)` 并在控制台打印平均耗时。
- 输出：`exp/exp2/output/coverage/...` 或 `exp/exp2/output/faithfulness/...`，按数据集和模型分目录。

使用说明
```bash
# 生成侧 RISE/MAS 忠实度
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
python exp/exp2/run_exp.py \
  --datasets exp/exp2/data/morehopqa.jsonl \
  --attr_funcs IG,perturbation_all,attention,perturbation_REAGENT,ifr_all_positions,perturbation_CLP,ifr_multi_hop,attnlrp,ft_attnlrp \
  --model qwen-8B \
  --model_path /opt/share/models/Qwen/Qwen3-8B/ \
  --cuda 0,1,2,3,4,5 \
  --num_examples 100 \
  --mode faithfulness_gen \
  --n_hops 3
```
常用参数：
- `--datasets`：逗号分隔数据集名；若已存在 `exp/exp2/data/<name>.jsonl` 则直接使用。
- `--attr_funcs`：逗号分隔方法（无 AT2）；`ifr_multi_hop` 与 `ft_attnlrp` 支持多跳（由 `--n_hops` 控制）。
- `--data_root`/`--output_root`：缓存与结果目录（默认 `exp/exp2/data` / `exp/exp2/output`）。
- `--mode`：faithfulness_gen；`--num_examples` 控制评测条数。 math 会被拒绝。***
