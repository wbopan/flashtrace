# FT 多跳案例分析 & IFR 标准可视化（exp/case_study）

此目录提供一个轻量的单样本 IFR 可视化流程，不改动核心评测代码。

## 功能
- 读取单个样本（默认 `exp/exp2/data/morehopqa.jsonl`，索引 0）。
- 支持多种模式：
  - `ft`：当前使用的多跳 FT 归因（内部调用 `LLMIFRAttribution.calculate_ifr_multi_hop`）。
  - `ifr`：标准 IFR（单 hop），默认对指定 sink span 做**聚合 IFR**（只显示 1 个面板）。
  - `attnlrp`：AttnLRP 的 sink-span 聚合（只显示 1 个面板，case study 额外记录裁剪前向量）。
  - `ft_attnlrp`：FT-attnLRP（多跳递归 AttnLRP，case study 额外记录每 hop 的裁剪前向量）。
- 可视化三个阶段：
  - **裁剪前 token 级**：带 chat template 的完整序列热力图。
  - **裁剪后 token 级**：去除模板后的用户输入 + 生成热力图，按 input/thinking/output 分段。
- 输出 JSON（完整数值）和 HTML（逐跳热力图）。
- 额外提供 MAS（faithfulness / token perturbation）可视化：对指定归因方法做 token 级扰动评估，并渲染扰动影响热力图 + MAS 分数。

## 快速开始
```bash
# 根据本地模型修改 model/model_path
# 多跳 FT（默认）
python exp/case_study/run_ifr_case.py \
  --mode ft \
  --dataset exp/exp2/data/morehopqa.jsonl \
  --index 0 \
  --model qwen-8B \
  --model_path /opt/share/models/Qwen/Qwen3-8B/ \
  --cuda 0 \
  --n_hops 1

# 标准 IFR（单 hop，可指定 sink span）
python exp/case_study/run_ifr_case.py \
  --mode ifr \
  --dataset exp/exp2/data/morehopqa.jsonl \
  --index 0 \
  --model qwen-8B \
  --model_path /opt/share/models/Qwen/Qwen3-8B/ \
  --cuda 0 \
  --sink_span 0 0

#（可选）IFR 按 sink_span 内每个生成 token 单独计算（会显示多个面板）
python exp/case_study/run_ifr_case.py \
  --mode ifr \
  --ifr_view per_token \
  --dataset exp/exp2/data/morehopqa.jsonl \
  --index 0 \
  --model qwen-8B \
  --model_path /opt/share/models/Qwen/Qwen3-8B/ \
  --cuda 0 \
  --sink_span 0 20

# AttnLRP（sink-span 聚合；默认 score_mode=max）
python exp/case_study/run_ifr_case.py \
  --mode attnlrp \
  --dataset exp/exp2/data/morehopqa.jsonl \
  --index 0 \
  --model qwen-8B \
  --model_path /opt/share/models/Qwen/Qwen3-8B/ \
  --cuda 0 \
  --sink_span 0 20

# FT-attnLRP（多跳递归 AttnLRP）
python exp/case_study/run_ifr_case.py \
  --mode ft_attnlrp \
  --dataset exp/exp2/data/morehopqa.jsonl \
  --index 0 \
  --model qwen-8B \
  --model_path /opt/share/models/Qwen/Qwen3-8B/ \
  --cuda 0,2,3,4,5,7 \
  --n_hops 3
```

产物位于 `exp/case_study/out/`，文件名前缀根据模式变化，例如：
- `ft_case_<dataset>_idx<idx>.json/html`
- `ifr_case_<dataset>_idx<idx>.json/html`
- `attnlrp_case_<dataset>_idx<idx>.json/html`
- `ft_attnlrp_case_<dataset>_idx<idx>.json/html`

## MAS（Faithfulness / Token Perturbation）可视化

> 说明：这里的 MAS 与项目 `llm_attr_eval.LLMAttributionEvaluator.faithfulness_test()` 保持一致：
> 1) 先对样本跑指定方法的归因，并取 token-level attribution（Seq / Row / Recursive）。
> 2) 按 prompt token 的重要性排序，逐步将 token id 替换为 `tokenizer.pad_token_id`（token 级扰动）。
> 3) 用 `sum log p(generation + EOS | prompt)` 得到分数曲线，计算 RISE / MAS / RISE+AP。
> 4) 可视化时用“每一步扰动带来的边际 logprob 变化”作为 token 分数，渲染为 token spans 的“扰动影响热力图”。

```bash
# FT-IFR（ifr_multi_hop；默认 --method ft）
python exp/case_study/run_mas_case.py \
  --dataset exp/exp2/data/morehopqa.jsonl \
  --index 0 \
  --model qwen-8B \
  --model_path /opt/share/models/Qwen/Qwen3-8B/ \
  --cuda 0 \
  --method ft \
  --n_hops 1
```

常用方法选择（与 `run_ifr_case.py` 的模式名对齐）：
```bash
# IFR（需要 sink_span；默认会优先使用数据集缓存字段）
python exp/case_study/run_mas_case.py --method ifr --sink_span 0 20 ...

# FT-IFR（ifr_multi_hop）
python exp/case_study/run_mas_case.py --method ft --n_hops 1 --sink_span 0 20 --thinking_span 0 20 ...

# AttnLRP（sink-span span-aggregate；默认会优先使用数据集缓存字段）
python exp/case_study/run_mas_case.py --method attnlrp --sink_span 0 20 ...

# FT-AttnLRP（attnlrp_aggregated_multi_hop）
python exp/case_study/run_mas_case.py --method ft_attnlrp --n_hops 1 --sink_span 0 20 --thinking_span 0 20 ...
```

产物位于 `exp/case_study/out/`，文件名前缀为：
- `mas_case_<method>_<dataset>_idx<idx>.json/html`

HTML 默认包含 3 个 attribution 视角面板（Seq / Row / Recursive），每个面板里有 2 行 token 级热力图：
- **Method attribution（token weights）**：该方法的 token 归因权重（用于排序/密度）。
- **Attribution-guided MAS marginal（path deltas）**：按归因排序逐步替换的边际影响（这就是评测中实际使用的扰动路径）。

## 在浏览器中查看 HTML
1) 先运行上面的命令生成 `.html`（终端会打印形如 `wrote exp/case_study/out/...html`）。

2) 在仓库根目录启动一个静态文件服务（任选一个端口，例如 8888）：
```bash
python -m http.server 8888 --directory exp/case_study/out
```

3) 用浏览器打开（注意是 `http://`，不是 `https://`）：
- 本机：`http://127.0.0.1:8888/<你的html文件名>`
- 远程机器（推荐端口转发）：在本地执行 `ssh -L 8888:127.0.0.1:8888 <user>@<server>`，然后在本地浏览器打开 `http://127.0.0.1:8888/<你的html文件名>`

如果你在 `http.server` 日志里看到大量 `400 Bad request version` 且伴随乱码，通常是有客户端用 HTTPS 去连了 HTTP 端口；请确认浏览器地址栏是 `http://...`。

## 可选参数
- `--sink_span a b` / `--thinking_span a b`：覆盖生成侧的 sink/thinking 句子 span（默认使用缓存字段）。
- `--ifr_view aggregate|per_token`：仅 `--mode ifr` 生效；`aggregate` 为 sink-span 聚合 IFR（默认 1 个面板），`per_token` 为逐 token（多面板）。
- `--lrp_score_mode max|generated`：仅 AttnLRP 模式生效；选择目标函数的打分方式。
- `--lrp_normalize_weights / --no-lrp_normalize_weights`：仅 AttnLRP 模式生效；是否对 sink 权重归一化。
  - 说明：case study 默认会优先用 bf16 加载模型以避免 fp16 下梯度下溢导致的全 0 归因（不影响其它模式）。
- `--chunk_tokens` / `--sink_chunk_tokens`：IFR 分块参数。
- `--output_dir`：修改输出目录。
- `--score_transform positive|abs|signed`：仅 `run_mas_case.py` 用于控制 token 热力图的显示方式（不改变 MAS 分数的计算逻辑）。

## 文件说明
- `run_ifr_case.py`：命令行入口与落盘（支持 `ft`/`ifr`/`attnlrp`/`ft_attnlrp` 模式）。
- `run_mas_case.py`：MAS（faithfulness / token perturbation）可视化入口与落盘（支持 `ifr`/`ft`/`attnlrp`/`ft_attnlrp`）。
- `analysis.py`：逐跳清洗与封装（token-level）。
- `viz.py`：HTML 渲染与热力图。
