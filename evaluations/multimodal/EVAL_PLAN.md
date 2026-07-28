# 视觉管线正式评估方案（v2，2026-07-24，主实验范围定稿）

本方案基于三份已有证据冻结：strict pilot（`results/strict/final/RESULTS.md`，Wiki n=18 / CLEVR n=20）、
native fit pilot（`results/strict/native_pilot/RESULTS.md`，三数据集各 n=10）、以及论文草稿
`paper/main.tex` 的视觉表格结构。

v2 相对 v1 的变化：按 2026-07-24 决定，只跑主实验（下表 E1–E5）。K=2 递归消融、appendix
方法（Rollout/Grad×Attention/TAM）、位置等变性重跑、效率 scaling、VISTAQA 新增运行、第二
VLM 全部裁剪；递归机制分析降为对 E3–E5 已有记录的离线分析。方案冻结后，样本选择、gate
定义、primary endpoint 不得再根据 attribution 结果调整。

## 0. 已冻结的顶层决定

| 数据集 | 最终角色 | 规模 | 位置 |
|---|---|---|---|
| Wiki-VISA | localization 主数据集（原生 element boxes） | strict n=120（三 strata 各 40） | 主表 |
| VizWiz-LF | 自然图像、长 OUTPUT 的 frozen-response faithfulness 主数据集 | n=100 | 主表 |
| VISTAQA | localization failure-analysis diagnostic | 不新跑；引用已有 native pilot n=10 | appendix |
| CLEVR-XAI | 单 token 输出、中心偏置的反例 diagnostic | 沿用已有 strict n=20，不扩量 | appendix |

- 模型：`Qwen/Qwen3-VL-8B-Thinking`，revision `92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b`。
- Wiki-VISA 固定 `max_pixels=2,007,040`（1MP 已被受控 OCR 检查否决，见 README）。
- 输入只有 `I_IMAGE + I_QUESTION`；THINKING/OUTPUT 全部模型自生成；dataset rationale、
  functional program 永不入 prompt。
- attribution sink 恒为完整 `OUTPUT_SPAN`；所有方法 teacher-force 同一条冻结响应。
- FlashTrace 主定义：direct `OUTPUT→IMAGE` + exact `OUTPUT→THINKING→IMAGE`，递归项按进入
  THINKING 的 attribution mass 比例加权（K=1）。`flashtrace-all-gen` 是消融，不是主方法。
- 文件分离：dataset / model / generation_eval / ablation.model / attribution records 各自独立
  JSONL，仅以 `sample_id` 连接。

正确性过滤原则：

- localization（Wiki-VISA）：whole-OUTPUT 正确是硬 gate。GT box 只对正确答案的证据有定义，
  答错样本上的 Recovery/Energy 会把模型错误误记为归因方法错误。
- faithfulness（VizWiz-LF）：不要求答案正确。faithfulness 度量归因图对模型自身冻结响应的
  忠实度，答错时 deletion/insertion 依然良定义。正确性只做软标注（fully/partial/wrong），
  用于 A8 的 fully-correct 子集敏感性检查。
- 全部过滤发生在 attribution 之前；funnel 表公开每步淘汰量；正确性筛选的代表性损失写进
  limitation。

## 1. 阶段 0：协议与基础设施冻结

1. 重写 `protocol.json`：删除旧的 COCO/GQA/RePOPE 设计，写入上表的数据集角色、规模、gate、
   endpoint、方法清单，并声明"冻结后不得改样本"。
2. `strict_datasets.py` 增加 `vizwiz-lf` manifest 生成（固定 seed，保留原生问题与 crowd 答案
   作为 evaluation metadata）。
3. VizWiz-LF strict gate 变体（无可用 exact-match，且不进 localization）：
   - 硬 gate：非拒答/非 unanswerable；两次 greedy 完全一致；generated vs teacher-forced
     token IDs 完全一致；global blur 后 `OUTPUT_SPAN` log-prob 下降；blur/gray generation 至少
     一个不能复现原 OUTPUT；THINKING 在 2048 token 内正常闭合；OUTPUT ≥ 16 tokens。
   - 软标注（只用于分层报告，不做筛选）：与 crowd 答案的一致性等级（fully/partial/wrong，
     LLM judge + 10% 人工复核）。
4. Wiki-VISA 沿用已验证的六条 strict gate（正确、稳定、token identity、blur log-prob 下降、
   ablation 非复现、THINKING 闭合）。
5. faithfulness 预算统一为约 64 region + 10 deletion/insertion steps + blur replacement
   （native pilot 的 VizWiz 用的是 36 region / 5 steps，正式运行必须按 64/10 重跑）。

## 2. 阶段 1：候选生成与 gate funnel（E1、E2）

- Wiki-VISA：每 stratum（first-page / later-page / non-passage）生成 80 个候选，共 240，
  从通过全部 gate 者中按固定 seed 取 40/40/40，锁定 120 个 `sample_id`。
- VizWiz-LF：生成 200 个候选，锁定 100 个，按 OUTPUT 长度三分位 + 问题类型分层记录。
- 必须产出完整 funnel 表：候选数 → 每条 gate 淘汰数 → 最终 n，写入 RESULTS 与论文 appendix。
- 人工审计：每数据集抽 10% 做 image-dependence 与 THINKING 质量审计，结论只作 caveat，
  不回头改样本。
- 样本 ID 锁定进 `results/strict/formal/frozen_ids.json`，后续一切实验只允许引用它。

## 3. 阶段 2：Wiki-VISA localization 主实验（E3）

- 方法（8）：Random、Center prior、Visual LOO、Visual IG、AttnLRP、FlashTrace(exact, K=1)、
  IFR-span（K=0 消融）、FlashTrace all-generation bridge（消融）。
  Rollout / Grad×Attention / TAM 已裁剪，不跑。
- 主指标（预注册 primary endpoint：Energy in evidence + Recovery@5%）：Energy、
  evidence rank AUC、R@5、R@20；从记录中可免费补 Pointing Game、Top-area IoU、R@1、R@10。
- 空间计分沿用 whole-patch tie-aware 协议（GT 像素映射到 patch，cutoff tie 期望记分；
  不用双线性平滑与 partial-patch top-q）。
- 报告 overall + 三 strata 分项；强调 Wiki boxes 是 supporting HTML element 而非词级 mask。
- 叙事按 pilot 校准：FlashTrace 强于 coverage/ranking，AttnLRP 强于 concentration，
  如实报，不宣称全面最好。

## 4. 阶段 3：frozen-response faithfulness 主实验（E4、E5）

- 主面板（E4）：VizWiz-LF n=100，统一 64-region / 10-step / blur 预算；冻结完整
  THINKING+OUTPUT，只累计 `OUTPUT_SPAN` log-prob。
- 次面板（E5）：Wiki-VISA n=120 复用同一批冻结样本跑同预算 faithfulness（零额外生成成本），
  入 appendix。
- 方法：同 E3 的 8 方法，主表与 appendix 均完整展示。
- 指标：Deletion AUC ↓（primary endpoint）、Insertion AUC ↑、Visual-MAS ↓；保存全部
  perturbation curves 与 degenerate-curve 计数。
- 符号敏感性：deletion/insertion 排序用 signed score、MAS 用 positive mass；另算一版
  positive-only 排序 sensitivity（纯分析，A4）。
- Center prior 必须整行在场：native pilot 中 center insertion AUC 0.704，FlashTrace 的
  任何优势必须显式超过 centered-subject prior。

## 5. 递归机制分析（纯分析，不新跑 GPU）

只用 E3–E5 已有记录：

1. K=0（IFR-span）vs K=1（主设定）vs all-generation bridge 的成对对比——这三个配置本身
   就在 E3–E5 的方法清单里；
2. 按 THINKING 长度三桶分层报告 K=1 相对 K=0 的增益；
3. attribution mass 在 direct 项与 recursive 项间的比例分布；
4. exact vs all-gen map 的 cosine 相似度（pilot 中 CLEVR 为 0.9969，检验长 THINKING 上
   是否分离）。

K=2 已裁剪，论文只声称"单跳递归的作用"，不外推更多 hop。

## 6. 偏置诊断（纯分析，不新跑 GPU）

- GT centroid 到图像中心的距离分布（Wiki 三 strata 与 VizWiz 各自报告）；
- heatmap border-mass ratio（strict pilot 已发现 CLEVR 上边界伪影，确认 Wiki/VizWiz 是否存在）；
- signed attribution vs positive-mass normalization 的指标敏感性；
- CLEVR 的 Unique First-nonempty vs Union 双 mask 结果沿用已有 n=20，不重跑。

位置等变性重跑已裁剪。

## 7. 统计与报告规范（全局）

- 一切方法对比只在共同成功样本 intersection 上进行；
- paired sample-level bootstrap ≥10,000 draws（沿用 pilot 的 50,000）；报告 95% CI、
  paired difference、W/T/L；
- primary endpoints 预注册：localization = Energy + R@5；faithfulness = Deletion AUC；
  Rank AUC 与 R@20 保留但注明易受中心/大区域偏置影响；
- CI 跨零就如实写"方向性证据"，不硬凑显著性；
- 每张主表配 funnel 表 + strata 分项；
- VizWiz fully-correct 子集敏感性检查（A8）入 appendix；
- n=18/20 strict pilot 与 n=10 native pilot 保留为协议验证与算力估计，不与正式结果合并汇总。

## 8. 论文改动清单（paper/main.tex）

1. `tab:visual_localization`（约 line 496）：三数据集列 → Wiki-VISA 单数据集，
   列为 Energy / Rank AUC / R@5 / R@20，行为 6 主方法 + 2 消融；strata 分项入 appendix；
2. faithfulness 表（约 line 521）：`CLEVR-XAI` 行组替换为 `VizWiz-LF`（主）；
   Wiki-VISA faithfulness 与 CLEVR-XAI 旧结果移 appendix；
3. 视觉协议段（line 490 placeholder）：写入 strict gates、revision、2MP、whole-patch 计分、
   64-region 预算、funnel 报告承诺；
4. appendix：VISTAQA 与 CLEVR 的 diagnostic 叙述引用已有 pilot 结果；mask-convention
   敏感性；偏置诊断；
5. 因裁剪产生的收缩：视觉效率 scaling 表不再规划（效率主张由文本侧实验支撑，视觉侧只引用
   E3–E5 记录中的实测 wall-clock/VRAM 作为佐证）；等变性、第二 VLM、K=2 相关表述全部删除；
6. limitations（line 713 placeholder）：视觉 mask 不完整性、VizWiz prompted 长输出、
   gate funnel 选择偏差、center prior、单模型单跳递归的范围限定。

## 9. 实验执行总表（定稿）

需要跑 GPU 的实验：

| # | 实验 | 数据集 × 规模 | 方法 | 关键产出 | 依赖 |
|---|---|---|---|---|---|
| E1 | 候选生成 + ablation audit | Wiki-VISA 240 候选 | generation + blur/gray ablation | funnel 表、锁定 120 IDs | 阶段 0 |
| E2 | 候选生成 + ablation audit | VizWiz-LF 200 候选 | 同上（VizWiz gate 变体） | funnel 表、锁定 100 IDs | 阶段 0 |
| E3 | localization 主表 | Wiki n=120 | 8 方法（6 主 + IFR-span + all-gen） | Energy、Rank AUC、R@5、R@20，overall + 3 strata | E1 |
| E4 | faithfulness 主面板 | VizWiz n=100，64 region / 10 步 | 同 8 方法 | Deletion AUC（primary）、Insertion AUC、Visual-MAS、curves | E2 |
| E5 | faithfulness 次面板 | Wiki n=120，同预算 | 同 8 方法 | 跨域一致性检查（appendix） | E1、复用 E3 冻结响应 |

纯分析项目（不新跑 GPU）：

| # | 项目 | 输入 | 产出 |
|---|---|---|---|
| A1 | THINKING 长度三桶分层的递归增益 | E3–E5 attribution records | 递归机制分析 |
| A2 | direct vs recursive mass 比例、exact vs all-gen cosine | 同上 | 递归机制分析 |
| A3 | GT centroid 中心距、border-mass ratio | E1/E2 manifest + heatmaps | 偏置诊断 |
| A4 | signed vs positive 归一化敏感性 | E3–E5 saved maps | 敏感性 appendix |
| A5 | paired bootstrap（50k）、W/T/L、funnel 汇总 | 全部 summary | 主表 CI 与统计报告 |
| A6 | CLEVR-XAI 反例叙述、双 mask 敏感性 | 已有 strict n=20 结果 | appendix，不重跑 |
| A7 | pilot timing 提取 → GPU-hours 排期 | pilot summary.json | 启动前算力预算 |
| A8 | VizWiz fully-correct 子集敏感性 | E4 结果 + 软标注 | faithfulness 稳健性 appendix |

已裁剪（2026-07-24 决定，主实验优先）：

| 原编号 | 内容 | 处置 |
|---|---|---|
| E6 | FlashTrace K=2 递归消融 | 不跑；递归分析降为 A1/A2，论文只声称单跳 |
| E7 | Rollout / Grad×Attention / TAM appendix 方法 | 不跑；appendix 方法表引用 strict pilot n=18/20 |
| E8 | 位置等变性重跑 | 不跑；偏置诊断保留纯分析部分（A3/A4） |
| E9 | 效率 scaling 网格 | 不跑；视觉效率佐证改用 E3–E5 实测 timing |
| E10 | VISTAQA 新增运行 | 不跑；appendix 引用 native pilot n=10 |
| E11 | 第二 VLM 泛化 | 不跑；写进 future work |

## 10. 执行顺序与算力

1. 阶段 0（协议冻结）→ 2. E1 ‖ E2（候选 + funnel + 锁 ID）→ 3. E3（先跑
   Random/Center/FlashTrace/AttnLRP/Visual IG 确认无异常，再补全 8 方法）→
4. E4 → 5. E5 → 6. A1–A8 分析与成文。

算力估计：正式规模约为 strict pilot 的 6–7 倍主样本量（38 → 220），外加 440 条候选
generation + ablation audit。启动前用 A7 从 `wiki_visa_n18_2mp_methods_v2/summary.json` 等
提取 per-sample per-method 实测时延与 VRAM 换算 GPU-hours；Grad×Attention 已裁剪，pilot 中
唯一 OOM 源随之消失。

## 11. 风险与止损

- VizWiz n=100 仍解析不出 FlashTrace vs AttnLRP 的 delta（pilot 差值约 +0.03）：如实报 CI，
  叙事落在"与最强 baseline 相当且预算恒定/更快"，不扩样本硬凑。
- Wiki funnel 产出率低于预期：扩候选池，绝不放松 gate。
- FlashTrace 在某 strata 或某指标弱于 baseline：保留在主表，配合偏置诊断做失效分析。
