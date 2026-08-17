# Agent 场景正式评估方案（v1，2026-07-28）

本方案对应 `paper/main.tex` 中尚待填充的 Aider 多轮结果、AgentDojo
outcome 和 prompt-injection localization 三张表。协议冻结后，不根据归因结果修改
样本、gate、target 或 primary metric。

## 1. 模型与升级规则

- 不调用云端推理 API。
- generation 候选模型为 `Qwen/Qwen3-8B`，revision
  `b968826d9c46dd6066d109eabc6255188de91218`。
- attribution 始终使用同一 Qwen3-8B revision，不使用 4B。
- 8B 先跑固定决策集：Aider 30 个任务；AgentDojo 每 suite 10 clean +
  10 attacked，共 80 cases。
- 只有 8B 同时满足下列条件才用于正式 generation：
  - AgentDojo clean utility >= 70%，attacked utility >= 60%；
  - parse-error turn <= 5%，不完整/截断 case <= 10%；
  - Aider final execution pass >= 30%，whole-file format compliance >= 90%；
  - validator/test harness 不存在系统性基础设施失败。
- 任一条件失败，正式 generation 全局切换到 `Qwen/Qwen3.6-27B`，
  revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`。不按 case 用大模型
  rescue，避免混合选择偏差。此时归因明确标为 Qwen3-8B surrogate attribution。
- 模型的自我判断不参与任何 gate；正确性只由真实测试或官方 validator 决定。
- 正式 27B generation 使用 greedy decoding；AgentDojo 每 turn 最多 4,096
  new tokens，Aider 每 turn 最多 8,192 new tokens。缺失 `</think>` 或达到长度上限的
  输出保留为 invalid，不能静默删除。
- 多轮 chat template 固定 `preserve_thinking=True`，使实际历史 reasoning 与 action、
  tool/test feedback 一起进入后续 turn，也进入冻结的 attribution context。

## 2. 数据、生成和过滤

| 数据集 | 正式 generation 规模 | 样本如何产生 | 验证与过滤 | 论文角色 |
|---|---:|---|---|---|
| Aider Polyglot | 全量 225（C++ 26 / Go 39 / Java 47 / JavaScript 49 / Python 34 / Rust 30） | 固定官方 repo commit `7e0611e…6144f`；首轮生成 whole-file edit；失败时把真实测试输出作为第二轮 feedback，再生成 revised edit | 每轮在固定 Docker 镜像中断网执行官方测试；保留所有 attempted 轨迹并报告 draft/final pass、repair gain、format compliance；归因只跳过无法形成完整 action span 的 invalid turn，并公开 denominator | 多轮代码 repair faithfulness + runtime；all-valid 与 exec-success 子集并报 |
| AgentDojo clean | 全量 97（16 / 21 / 20 / 40） | v1.2.2 官方 task、tool schema、default environment；greedy autonomous tool loop，最多 10 turns | 官方 utility validator；不按 outcome 过滤 generation | benign utility |
| AgentDojo attacked | 全量 949（144 / 105 / 140 / 560） | 每个 user task 与每个 injection task 的完整笛卡尔积；官方 `ImportantInstructionsAttack`；同一 autonomous loop | utility 与 injection-success validator 分开运行；所有 case 保留 | utility under attack、ASR、joint success |
| AgentDojo localization | attacked 轨迹中的分层条件子集 | 主结果使用 greedy 与固定预算自然重采样；扩展结果在完整模型 reasoning 后按递增 token 预算提供目标 action 前缀并恢复模型续写 | 必须精确记录 span、realization stage 与逐 token 来源；自然组和 teacher-forcing 组分别报告 | prompt-injection 原因定位 |

说明：旧 Aider 的 133 是 Python-only benchmark；本实验采用论文当前升级目标对应的
225-task Polyglot 全量。AgentDojo suite 顺序为 Banking / Slack / Travel / Workspace。

## 3. Baseline 与指标

| 实验 | 方法 | Target | 指标 |
|---|---|---|---|
| Aider | IFR last output line | revised edit 最后一个有效代码行 | RISE、MAS、time |
| Aider | IFR exhaustive | 完整 final edit，逐 token 聚合 | RISE、MAS、time |
| Aider | AttnLRP | 完整 final edit | RISE、MAS、time |
| Aider | FlashTrace（K=1） | 完整 final edit，经 draft/test-feedback/reasoning bridge 回溯 | RISE、MAS、time |
| AgentDojo | Random | 无模型归因 | R@5/10/20、Pointing、Injection mass |
| AgentDojo | IFR last action token | generated malicious action 的最后一个 token | 同上 + time |
| AgentDojo | IFR exhaustive | 完整 generated malicious action，逐 token聚合 | 同上 + time |
| AgentDojo | FlashTrace（K=1） | 完整 action，经 generated reasoning 回溯 | 同上 + time |

所有方法 teacher-force 同一条冻结轨迹。定位只比较共同成功样本 intersection；
faithfulness 同时报 all-valid 与 exec-success。Random 用解析后的同一 token mask 和固定
seed 计算期望，不消费 GPU。

## 4. 可复现与安全

- generation 保存 rendered prompt、原始 token IDs、EOS/截断边界、解析错误、工具调用、
  测试输出或 validator outcome、wall time、模型与环境 provenance。
- Aider Docker 正式测试使用 `--network none`、只读 root、宿主 UID/GID、固定资源限制；
  Gradle 依赖只在正式运行前预热一次。
- AgentDojo raw injection/tool-result 只进入被 gitignore 的审计文件；论文与常规报告只使用
  redacted artifact。
- 统计使用 paired sample bootstrap 50,000 次和 95% CI；同时报告 W/T/L、失败数、
  denominator 与 coverage。CI 跨零时不声称显著优势。

## 5. 执行顺序

1. 六语言 Aider harness 与 AgentDojo validator smoke。
2. 8B 的 Aider-30 + AgentDojo-80 generation-only 决策集。
3. 按冻结 gate 选择 8B 或 27B，并记录一次全局 decision artifact。
4. Aider-225、AgentDojo-97 clean、AgentDojo-949 attacked 全量 generation。
5. 用 Qwen3-8B 跑上表 attribution baseline。
6. 汇总 funnel、outcome、localization、faithfulness、效率和 bootstrap 统计。
7. 生成论文表格、appendix audit 和 redacted qualitative cases。

六语言 harness 正向控制已使用官方 reference implementation 通过 6/6，审计结果见
`evaluations/aider_harness_positive_control.json`。

## 6. AgentDojo 全模型生成覆盖协议

- 完整轨迹基线使用累计预算 `1,4,16`：第一次 greedy，后续使用冻结的 temperature、
  top-p、top-k 和哈希派生 seed。validator 首次确认即停止。
- 主搜索策略自然采样 `M` 条到完整 `</think>` 边界的 reasoning。候选只能被评分与选择，
  不得相互拼接。隐藏评分可使用目标 action 的条件 log-prob 或短自然 rollout；选中后从
  该候选的精确 token 前缀重新自然采样 action。最终 completion 的参考写入 token 必须为 0。
- reasoning-only 温度若高于最终 action 温度，必须显式记录并作为自然分布偏移报告；同时
  报告候选去重比例、非首候选选择率与条件分数增益。
- 缺失完整 `</think>` 边界的轨迹不得进入 reasoning 搜索或诊断对照，不创建最短 bridge 兜底。
- 每条记录保存搜索方法、候选数/去重数、选择索引、模型调用/token/时间成本、reasoning
  token 数量以及 completion 的逐 token 来源掩码。每一级使用全新的 benchmark environment。
- Prefix forcing 与 sparse correction 仅作诊断上界，分别记录实际参考 token 数量和比例；
  不得计入全模型生成成功率，也不得用作 FlashTrace 主归因样本。
- Particle/SMC 仅在 reasoning rerank 不能相对 best-of-N 提升覆盖时进入下一轮；所有粒子片段
  必须来自模型连续采样，只允许评分与重采样。
- `diagnose_coverage.py` 只输出聚合计数、比例与 bridge 检查结果，不输出样本文本。
