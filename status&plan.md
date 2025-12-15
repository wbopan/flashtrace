# FlashTrace / CAGE 风格归因与 MAS/RISE：当前实现状态 & Token-level 改进计划（给外部专家评估用）

> 本文档面向**无法访问本仓库代码**的外部算法专家，尽量完整描述：  
> 1) 当前项目“归因（attribution）”与“MAS/RISE 忠实度评估（faithfulness / perturbation）”的算法实现逻辑（当前是 sentence-level）；  
> 2) 我们计划把两者升级为 token-level 的动机、定义、实现可行性与主要风险；  
> 3) 需要专家评估的两个核心问题（是否基于 CAGE、是否存在 token-level 对应实现/先例与推荐做法）。
>
> 说明：本文只做调研/分析与改进方案设计，不做任何代码改动。

---

## 0. 项目语境与目标（你需要知道的最少背景）

### 0.1 项目实现了什么

当前项目已经实现多种归因方法，并在两个实验目录里提供批量评测与可视化：

- 归因方法（示例，不限于）：`IFR`、`FT-IFR (multi-hop IFR)`、`AttnLRP`、`FT-AttnLRP`、`IG (Integrated Gradients)`、`Attention`、`Feature Ablation` 等。
- 扰动式忠实度评估指标：`RISE`、`MAS`、`RISE+AP`（三者一起输出）。
- 批量评测入口：`exp/exp2/`（采样+过滤样本、批量跑归因与评估）。
- 案例可视化入口：`exp/case_study/`（对归因与 MAS/RISE 过程做更细粒度可视化）。

### 0.2 样本结构（input / think / output）

项目在 `exp/exp2/` 的采样/过滤阶段，会得到包含三段语义的样本：

- **input**：用户输入（上下文+问题），进入 prompt template 与 chat template 后成为模型条件。
- **think**：模型生成的 CoT/推理文本（作为 generation 的一段子 span）。
- **output**：模型最终答案（作为 generation 的一段子 span，通常在最后）。

采样产物通常会记录：

- `target`：拼接后的 generation 文本（think + output，且通常会补一个 eos 用于打分）。
- `thinking_span` / `sink_span`：在 generation token 序列中的 token 索引区间，用于 multi-hop 方法把“output 归因先投到 think，再投到 input”。

### 0.3 我们现在要做的“重大算法改进”

现有实现里，“归因输出”和“MAS/RISE 评估”都以**句子（sentence）**作为基本单元（feature / perturbation unit）。  
我们希望把两者升级为**token-level**（以 tokenizer token 为单元），以便更细粒度分析：

1) baseline 方法的归因质量 vs FT-IFR / FT-AttnLRP 每一跳（hop）归因效果差异；  
2) `MAS/RISE` 在评估不同方法/不同 hop 的归因结果时，是否“区分得开”、是否稳定、是否存在句子聚合带来的信息损失/偏置。

---

## 1. 当前实现：从“模型输入/输出”到“句子级归因”再到“MAS/RISE”

为避免歧义，本节用“坐标系 + 张量形状 + 算法定义”描述当前实现。

### 1.1 Prompt 与 Chat Template（固定背景）

当前项目对用户输入 `prompt` 会先套一个固定的 prompt template，再套模型 chat template：

- prompt template（固定字符串）：
  - `DEFAULT_PROMPT_TEMPLATE = "Context:{context}\n\n\nQuery: {query}"`
  - 通常 `query=""`，仅用 `context=prompt`
- chat template：
  - 使用 `tokenizer.apply_chat_template([...], tokenize=False, add_generation_prompt=True, enable_thinking=False)`

因此模型真正看到的“完整 prompt token 序列”包含三部分：

1) chat template tokens（系统/角色/分隔符等）  
2) user prompt tokens（用户输入文本经 prompt template 包裹后，在 chat template 中占据一段连续 token 区间）  
3) generation tokens（模型生成 tokens，通常包含 think + output）

**重要：**后续所有“扰动评估”只对 **user prompt** 做替换；chat template tokens 永远不被替换（相当于常量背景）。

### 1.2 Token-level 归因矩阵：形状与裁剪（trim chat template columns）

多数归因方法首先在 token 级别得到一个归因矩阵（或向量），其核心对齐方式是：

- 行（rows）= sink tokens：生成侧 token（哪些输出 token 被解释）
- 列（cols）= source tokens：输入侧 token（哪些 token 被认为影响了输出）

典型的 token-level 归因矩阵形状：

- `A_full ∈ R^{G × (P_full + G)}`
  - `G`：generation token 数
  - `P_full`：*完整 prompt* token 数（含 chat template + user prompt）
  - 列包含两段：
    - prompt（含 chat template + user prompt）
    - generation（允许对“过去生成 token”归因）

之后会做一次统一的 **trim**：删除 chat template 对应的列，只保留 user prompt 与 generation 的列：

- `A_trim ∈ R^{G × (P_user + G)}`
  - `P_user`：user prompt token 数

trim 的概念实现：

1) 在完整 prompt token 序列里定位 user prompt 占据的连续区间 `[i, i+P_user-1]`；  
2) “保留列索引” = user prompt 的列 + generation 的列；其它列（chat template）全部删除。

**结论：**当前项目“先对带 chat template 的完整序列做归因，再裁剪掉 chat template token 的列”是确定的。

### 1.3 Token → Sentence 聚合：句子级归因矩阵 `S`

当前项目会把 token-level 归因矩阵 `A_trim` 聚合到 sentence-level，得到句子级归因矩阵：

- prompt 句子数：`N_p`
- generation 句子数：`N_g`（生成里会把 `eos_token` 单独当成一个句子，便于对齐/可视化）

句子级归因矩阵：

- `S ∈ R^{N_g × (N_p + N_g)}`
  - 行（rows）= generation sentences
  - 列（cols）= prompt sentences + generation sentences

聚合规则（核心就是“块求和”）：

- 对任意一对（生成句子 i，来源句子 j）：
  - 取出 `A_trim` 中属于“生成句子 i 的 token 行集合”与“来源句子 j 的 token 列集合”的子矩阵；
  - 对该子矩阵求和，得到 `S[i,j]`。

关键数值处理：

- NaN → 0
- 负归因 → clamp 到 0（只保留正归因）
- 行归一化（每行和为 1）

> 直觉：这一步把 token-level 归因“平滑/压缩”到句子级，便于排序与 perturbation。

### 1.4 Seq / Row / Rec（三种句子级归因视角）

项目对外主要使用三种句子级归因视角（评测 CSV 与 case study HTML 都展示）：

1) **Seq attribution（seq）**：直接使用 `S`（形状 `N_g × (N_p+N_g)`）
2) **Row attribution（row）**：把若干“需要解释的输出行”相加得到一个向量  
   - `row ∈ R^{1 × (N_p+N_g)}`
3) **Recursive attribution（rec）**：CAGE 风格的“回卷/展开”  
   - `rec ∈ R^{1 × (N_p+N_g)}`

#### 1.4.1 CAGE 风格回卷（recursive / rec）的算法定义（当前是 sentence-level）

rec 的动机：后面的生成可能依赖前面的生成（尤其含 CoT）。只看某一行 `S[t,:]` 可能低估“经中间生成句间接传递”的影响；因此把“归因到中间生成句子”的那部分进一步沿生成链条展开回 prompt。

对要解释的目标句子 `t`，回卷过程相当于：

1) 初始化：`r = S[t, :]`
2) 依次对更早生成句子 `k = t-1, t-2, ...`：
   - 用当前 `r` 里“对生成句子 k 的权重”作为系数，乘上 `S[k, :]` 并加回 `r`
   - 然后把 `r` 里“对生成句子 k 的权重”清零（表示已展开）

这就是当前实现中被命名为 `compute_CAGE_sentence_attr(...)` 的逻辑。

### 1.5 MAS/RISE（Faithfulness / Sentence Perturbation）评估：算法定义

MAS/RISE 的输入是：

- 句子级 attribution（通常只取 prompt-side 的列，即 `N_p` 列）
- prompt 的句子分割 `prompt_sentences[0..N_p-1]`
- 固定的 generation 文本 `generation`（可能包含 think + output）

#### 1.5.1 扰动单位与扰动方式（当前：sentence-level）

扰动单位：**prompt 句子**。

每一步选择句子 idx 后：

- 取该句子的 token 数 `n_tok = len(tokenizer(sentence).input_ids)`  
- 用 `eos_token` 字符串重复 `n_tok` 次替换整句：`sentence := eos_token * n_tok`

直觉：保持替换后的 token 数大致一致，减少“长度变化”带来的额外影响；但本质仍是“整句替换”，不是逐 token 删除。

#### 1.5.2 打分函数（score）

每一步扰动都会得到一个新 prompt（会重新套 chat template），然后计算：

- `score(prompt) = Σ_t log p( generation + eos | prompt )`

也就是：固定 generation，把 prompt 当条件，计算生成整段 generation（加一个 eos）的对数似然之和。

#### 1.5.3 归因引导的删除路径（guided deletion path）

先对 prompt 句子按归因从大到小排序：

- `w_j = Σ_rows attribution[row, j]`（对所有输出行求和）
- 得到降序排列 `π(0), π(1), ...`

然后沿该顺序逐句替换并记录 `scores[0..N_p]`：

- `scores[0]`：原始 prompt 的 score
- `scores[k]`：替换了前 k 句（最重要的 k 句）后的 score

同时构造“密度曲线” `density[0..N_p]`：

- `density[0] = 1`
- `density[k+1] = density[k] - w_{π(k)} / Σ_j w_j`

#### 1.5.4 三个指标（RISE / MAS / RISE+AP）

对 `scores` 做 monotonic normalization：

- 先线性归一化到 `[0,1]`（以 `scores[0]` 与 `scores[-1]` 定标）
- 再强制单调不增（running minimum）
  - 得到 `normalized_model_response[0..N_p]`

alignment penalty：

- `AP[k] = | normalized_model_response[k] - density[k] |`

输出三个 AUC：

- `RISE = AUC(normalized_model_response)`
- `RISE+AP = AUC(normalized_model_response + AP)`
- `MAS = AUC(corrected_scores)`，其中 `corrected_scores` 为：
  - `corrected_scores = normalized_model_response + AP`
  - 对其做 `clip(0,1)` 与再次 `0-1` 归一化（避免越界导致“虚假变好”）

---

## 2. 当前 sentence-level 方案是否“基于已有工作”？是否就是 CAGE？

### 2.1 仓库内可直接证明的证据（不依赖互联网）

从仓库现状可以非常强地推断：当前 sentence-level 的归因与评估方案整体来自/复刻了 CAGE 风格设计，证据包括：

1) 工程名：`pyproject.toml` 里 `[project].name = "cage"`  
2) 代码显式命名：
   - 存在 `compute_CAGE_sentence_attr(...)`，并把 rec attribution 称为 CAGE 风格回卷/展开
   - `ifr_core.py` 的模块级 docstring：`"IFR utilities integrated for CAGE"`
3) 指标命名与公式形态：
   - `faithfulness` 输出列名就是 `RISE, MAS, RISE+AP`
   - `alignment_penalty` 的用法（把模型响应曲线与 density 曲线对齐）非常“特定”，不像常见的 AOPC/Deletion/Insertion 直接替代

### 2.2 仍建议专家用互联网核对的点

尽管“CAGE 风格”几乎可以确定，但仍建议专家核对两点：

1) 原工作中默认 feature unit 是否确实就是 sentence-level（而非 token/span/word），以及其设计动机（理论/实验/成本）。  
2) `RISE/MAS/RISE+AP` 的正式定义是否与本文第 1.5 节完全一致（特别是：替换 baseline、monotonic normalization、density 与 AP 形式）。

---

## 3. 当前项目或相关工作里是否已有 token-level 的归因与 MAS/RISE？

### 3.1 在当前项目里：token-level 归因“有底层产物”，token-level MAS/RISE“没有主路径”

可确认现状：

- token-level 归因的底层产物存在：多数方法都会得到 `A_trim`（行是输出 token、列是输入 token）。  
- 但进入评测与主要分析口径时，会统一：
  - token → sentence 聚合得到 `S`
  - 再做 seq/row/rec
  - MAS/RISE 基于 prompt 句子扰动路径

因此：

- token-level 归因目前更多用于可视化/中间分析；主评测口径仍是 sentence-level。  
- token-level MAS/RISE 评估目前没有实现。

### 3.2 在 CAGE/相关工作里：token-level 版本是否存在，需要专家确认

我们无法在无互联网条件下 100% 确认 CAGE 原论文/原代码是否提供 token-level 版本，但从常识与现有实现推测：

- CAGE 很可能以 sentence-level 作为默认 feature unit（可读性、成本可控、降噪）。  
- token-level 的 RISE/MAS 理论上可定义，但可能因成本/噪声/扰动不自然等原因不作为主实验口径。

> 这正是我们希望专家评估的重点：  
> “如果把 sentence 换成 token，是否仍然是一个合理/可比较/可复现的评价体系？是否有先例或推荐做法？”

---

## 4. 从 sentence-level → token-level：算法层面的可行方案（请专家评估）

本节给出一个“严格对齐当前定义”的 token-level 改造方案：尽可能不改变指标语义，只把 feature 单元从 sentence 改为 token。

### 4.1 Token-level 归因：我们希望最终对外暴露什么

我们希望把当前三视角（seq/row/rec）升级为 token-level：

#### 4.1.1 Token-level `seq`

句子级 `seq` 当前是 `S ∈ R^{N_g × (N_p+N_g)}`。token 级对应为：

- `seq_token = A_trim ∈ R^{G × (P_user+G)}`

#### 4.1.2 Token-level `row`

句子级 row 是“把 indices_to_explain 的几行相加”。token 级可直接复刻：

- 选择要解释的输出 token 行集合 `T`（例如 sink span 对应 token 行）
- 定义：
  - `row_token = Σ_{t∈T} A_trim[t, :]`，得到 `1 × (P_user+G)` 向量

#### 4.1.3 Token-level `rec`（token 化的 CAGE 回卷）

句子级 rec 的本质是“把归因到中间生成节点的那部分沿生成链条展开回 prompt”。token 级可以用同样递推，只是把中间节点从生成句子换为生成 token。

设：

- `A_pp[t]`：输出 token `t` 对 prompt token 的直接归因向量（长度 `P_user`）
- `A_pg[t,k]`：输出 token `t` 对过去生成 token `k` 的直接归因权重（`k<t`）

递推：

- `R[t] = A_pp[t] + Σ_{k<t} A_pg[t,k] * R[k]`

该系统是严格下三角，可用前向递推或 triangular solve 实现。

> 需要专家评估：  
> 1) token-level rec 是否仍然合理、是否会因 token 过多导致噪声/数值问题；  
> 2) 是否仍建议“只保留正归因并归一化”（否则 density 与 AP 的语义可能不成立）。

### 4.2 Token-level MAS/RISE：把 perturbation unit 从 sentence 改为 token

希望尽可能保持当前 MAS/RISE 语义：

- guided deletion path（按归因从大到小）
- 替换为 baseline token（当前句子级用 eos_token 重复）
- score 仍为 `Σ log p(generation+eos | prompt)`
- density + alignment penalty + 三个 AUC 不变

唯一变化：`segmented_prompt` 不再是句子列表，而是 token 列表。

#### 4.2.1 Token-level prompt segmentation（关键工程/算法点）

要把 prompt 拆成 token list，且尽量满足：

- `"".join(tokens) == original_prompt_text`（可逆拼接）
- 每个 token 对应 tokenizer 的一个 token id（或至少稳定映射）

当前项目已存在一个可用方向：用 tokenizer offset mapping 把文本切成与 tokenizer 对齐的 text spans，这类 token 文本片段拼接后能还原原文。

> 需要专家评估：  
> token-level perturbation 的“token”应是 BPE/subword token、word token、还是更大粒度 span？  
> 若用 subword token 替换，扰动很不自然（例如把一个词拆开替换其中 1/3），是否会导致 MAS/RISE 失真？

#### 4.2.2 Token-level replacement baseline（关键语义点）

sentence-level 当前用 `eos_token` 重复 `n_tok` 替换整句。token-level 最直接是一对一替换：

- 将被选中的 prompt token 替换为一个 baseline token（例如 eos_token 的文本形式）

但 token-level 替换会引入新选择题：

- baseline token 用什么更合理？
  - eos_token：实现简单，但在 prompt 中大量出现可能产生非自然模式；不同模型处理也可能不同。
  - pad/unk/mask：对 causal/chat LM 未必语义稳定或可用。
  - 随机 token：噪声更大，但可能更接近 “remove information”。
- 是否需要保持 token 数恒定？
  - token-level 理论上一对一；但如果 replacement 文本导致重新分词成多个 token，会破坏一对一对应与密度定义。

> 需要专家评估：CAGE/相关工作通常用什么 baseline？token-level 是否推荐另选 baseline？

#### 4.2.3 复杂度与可运行性：token-level MAS/RISE 是否算得动

sentence-level MAS/RISE 需要 `O(N_p)` 次 score。token-level 会把 `N_p` 变成 `P_user`（prompt token 数），可能从几十变成几百/几千甚至上万（长上下文）。

因此我们预期需要至少一种降本策略（否则在长上下文不可用），可能包括：

1) **Top-K token only**：只扰动归因最高的 K 个 token（例如 K=50/100/200），其余视为尾部，用插值近似 AUC。  
2) **Chunked token**：把 token 分成固定长度块（例如 4/8/16 tokens）作为 perturbation unit（介于 sentence 与 token 之间）。  
3) **Random mask / RISE-style sampling**：借鉴图像 RISE 思路，用随机 mask 采样估计重要性/曲线。  
4) **Batch score**：每一步 perturbation 形成一批 prompts，批量算 logprob（受显存限制）。

> 需要专家评估：如果坚持“严格 token-level”，通常用哪种降本方式更被社区接受/更可发表？

---

## 5. 我们希望专家评估的两个核心问题（请直接给结论/建议）

### Q1：当前 sentence-level 的归因与 MAS/RISE 实现，是否确实就是基于 CAGE？有哪些关键偏差？

请专家确认：

1) 本文第 1 节定义与 CAGE 原论文/原代码在以下方面是否一致：  
   - feature unit（sentence）  
   - recursive attribution（回卷/展开）  
   - RISE/MAS/RISE+AP 的公式、monotonic normalization、alignment penalty  
   - eos 替换策略  
   - score 定义（固定 generation 的 log-likelihood）  
2) 若存在偏差：哪些偏差会显著影响“评估结论”的有效性？是否需要修正口径？

### Q2：CAGE 或相关工作里是否存在 token-level 的“归因 + MAS/RISE 评估”实现？如果没有，token-level 改造是否值得做、应如何做才合理？

请专家给出：

1) 先例：是否有工作在生成式解释评估里使用 token-level deletion/perturbation + logprob AUC（或等价指标）？  
2) 如果做 token-level MAS/RISE：  
   - token 定义选 subword / word / span 哪个更合理？  
   - baseline token 选什么更合理？  
   - rec（回卷）在 token-level 是否建议保留？还是只做 row/seq？  
   - 为可运行性，社区更接受哪些近似/降本策略？

---

## 6. 建议的实施路线图（供专家判断工程投入 vs 研究价值）

> 下面是内部的初步路线图，用于让专家评估“改造工程量与研究收益是否匹配”。

### Phase A：定义与对齐（最重要）

1) 明确 token unit（subword / word / span）与对齐方式  
2) 明确 token-level attribution 三视角定义（seq/row/rec 是否都保留）  
3) 明确 token-level MAS/RISE 的 baseline token 与是否保持 token 数恒定  
4) 明确是否同时保留 sentence-level 口径作为对照（建议至少保留一段时间）

### Phase B：实现 token-level MAS/RISE（带降本开关）

1) 实现严格 token-level（reference implementation）  
2) 实现 Top-K / chunk / batch 等可运行模式  
3) 做最小可复现实验：短 prompt、小模型、小样本，先验证曲线与指标合理性

### Phase C：迁移分析与可视化

1) baseline vs FT-* 每 hop 的 token-level 对比（含 per-hop attribution vector）  
2) token-level MAS/RISE 曲线可视化（对照现有 sentence-level HTML）  
3) 总结 sentence-level 与 token-level 差异：是否出现“句子聚合掩盖 hop/方法差异”的现象

---

## 7. 建议补充给专家的材料清单

为方便专家评估，建议（除本文外）再附带：

1) 一页“当前实现的数学定义”摘要（可直接引用第 1 节）  
2) 一页“token-level 改造后的数学定义”摘要（第 4 节）  
3) 2~3 个样例的可视化截图（sentence-level 与 token-level 对照，尤其是 multi-hop 每一跳）  
4) 粗略成本估算（以某个模型/上下文长度为例：sentence-level vs token-level 的打分次数与预计耗时）

---

## 8. 结论（供专家快速判断）

- 当前仓库的归因聚合、recursive attribution（回卷）以及 RISE/MAS/RISE+AP 评估形式，**几乎可以确定是 CAGE 风格体系**（工程名/函数名/指标名都指向这一点）。  
- 在当前项目里，token-level 归因底层产物已经存在（只是主流程把它聚合成 sentence-level）；但 token-level 的 MAS/RISE 尚未实现。  
- 从算法定义上，token-level 改造是可行的（第 4 节给出严格对齐现有定义的版本），但主要风险是：  
  - 计算成本爆炸（扰动步数从句子数→token 数）  
  - subword token 扰动语义不自然，可能导致评估口径失真  
  - token-level 回卷（rec）可能引入更强噪声/数值不稳定  

希望专家从“研究价值/可发表性/可复现性/成本收益比”的角度判断 token-level 方案是否值得推进，以及应采用哪种 token/word/span 定义与降本策略。
