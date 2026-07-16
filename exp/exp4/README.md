# FlashTrace 实验 4：Aider 多轮智能体轨迹归因

本目录把 ICML 版本的单轮 Aider 代码生成实验升级为多轮 repair trajectory benchmark，同时保留旧数据和旧结果格式的兼容性。

主要文件：

- `sample_and_filter.py`：通过 OpenAI 兼容 API 构造多轮轨迹，并用 judge 过滤最终答案。
- `trajectory_utils.py`：多轮 schema、校验、Qwen chat-template 渲染和 legacy 兼容层。
- `run_exp.py`：IFR/FlashTrace 归因、RISE/MAS，以及逐 token/逐 turn 轨迹归因输出。

## 1. Benchmark 构造

### 输入种子

种子沿用原 `exp4` 格式，每行一个 JSON：

```json
{"input": "用户指令与代码 stub", "output": "参考代码编辑", "length": 123}
```

默认路径为 `exp/exp4/data/aider.jsonl`。其中 `output` 只提供给 feedback/judge 模型，不会直接放进生成模型的上下文。

### 多轮协议

默认生成两次 assistant 编辑：

1. generator 根据 Aider 任务生成首轮 `draft_edit`；
2. judge 对照参考输出，生成不泄露参考代码的、类似单测失败信息的反馈；
3. 反馈作为新的 user turn 送回 generator，生成 `revised_edit`；
4. judge 对最终编辑做 True/False 判定，仅保留 True 轨迹。

`--assistant_turns N` 可扩展为更多轮，每个中间 assistant turn 后都会插入一轮 feedback。该实现不执行模型生成的代码；反馈和最终正确性均为 LLM judge 判定，这是相对于 Aider 原始可执行测试 harness 的明确限制。

### 输出 schema

`aider_multiturn.jsonl` 每行结构如下：

```json
{
  "schema_version": 2,
  "benchmark": "aider_multiturn",
  "id": "sample-id",
  "messages": [
    {"role": "system", "content": "...", "kind": "agent_instruction"},
    {"role": "user", "content": "...", "kind": "task"},
    {"role": "assistant", "content": "...", "kind": "draft_edit"},
    {"role": "user", "content": "...", "kind": "test_feedback"},
    {"role": "assistant", "content": "...", "kind": "revised_edit"}
  ],
  "metadata": {
    "generator_model": "qwen3-235b-a22b-2507",
    "judge_model": "deepseek-v3-1-terminus",
    "assistant_turns": 2,
    "judge_response": "True"
  }
}
```

数据本身不固化任何 tokenizer 特殊 token。归因运行时，`trajectory_utils.load_aider` 使用目标 Qwen3 tokenizer 的官方 chat template 渲染 `messages[:-1]`，并把最后一个 assistant message 作为 attribution target。因此同一条轨迹可由不同 Qwen3 checkpoint 评测。

### 采样命令

```bash
export FLASHTRACE_API_KEY=...

python exp/exp4/sample_and_filter.py \
  --seed_path exp/exp4/data/aider.jsonl \
  --out exp/exp4/data/aider_multiturn.jsonl \
  --max_examples 100 \
  --assistant_turns 2 \
  --api_base http://localhost:4000/v1 \
  --generator_model qwen3-235b-a22b-2507 \
  --judge_model deepseek-v3-1-terminus
```

API key 的读取顺序为 `--api_key`、`FLASHTRACE_API_KEY`、`OPENAI_API_KEY`。脚本支持 HTTP 429 `Retry-After`、普通请求错误重试、请求节流和端点 cache 参数；行为与 `exp2/sample_and_filter.py` 一致。

## 2. 轨迹归因

### Attribution 切片

对于 schema-v2 轨迹：

- prompt：最终 assistant turn 之前的完整 system/user/assistant 历史，使用目标 tokenizer 的官方 chat template；
- target：最终 assistant turn 的代码编辑；
- prompt segments：每个历史 message content 的字符 span，并通过 offset mapping 转成 token span；
- sink：保持原 `exp4` 协议。

方法和 sink：

- `ifr_all_positions / last_line`：最终编辑中最后一个非空、非 fence 代码行；
- `ifr_all_positions / last_token`：上述代码行的最后一个 token；
- `ifr_multi_hop_both / full_output`：完整最终编辑，排除框架追加的 EOS。

忠实度扰动直接复用归因阶段的精确 prompt 和 `user_prompt_indices`。这对多轮 chat template 很重要，可避免把 attribution token 位置错误地应用到另一层 prompt wrapper。

### 运行命令

```bash
python exp/exp4/run_exp.py \
  --data_path exp/exp4/data/aider_multiturn.jsonl \
  --output_root exp/exp4/output \
  --model qwen-8B \
  --model_path /opt/share/models/Qwen/Qwen3-8B/ \
  --cuda 2,3,4,5,6,7 \
  --num_examples 100 \
  --n_hops 3 \
  --k 20 \
  --save_trajectory_traces
```

`--no-save_trajectory_traces` 可关闭样本级轨迹输出。旧 `{input, output}` 数据仍可直接传入 `run_exp.py`，并保持 legacy raw prompt 行为。

### 输出

聚合忠实度 CSV：

```text
exp/exp4/output/faithfulness/aider/<model_tag>/row_only_<N>_examples.csv
```

列为：

```text
Method,Sink,Row_RISE_Mean,Row_RISE_Std,Row_MAS_Mean,Row_MAS_Std,Used,Skipped,Avg_Sample_Time_s
```

多轮轨迹 attribution JSONL：

```text
exp/exp4/output/faithfulness/aider/<model_tag>/trajectory_attribution_<N>_examples.jsonl
```

每个成功的 `method/sink/sample` 记录包含：

- `prompt_tokens`、`target_tokens`；
- `prompt_token_attribution`；
- 每个历史 message 的 role/kind/turn、char span、token span、mass 和归一化 fraction；
- chat template 结构 token 的 `unassigned_mass`；
- FlashTrace 每一 hop 的逐 turn attribution mass。

## 3. 本地验证

不需要 API key 的测试：

```bash
pytest -q tests/test_exp4_multiturn.py
```

测试覆盖 legacy/new schema、官方 chat-template 渲染、多轮生成编排、judge 过滤，以及随机初始化 tiny Qwen3 上的一次真实 `ifr_multi_hop_both` 轨迹归因前向计算。
