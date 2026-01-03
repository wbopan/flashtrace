# FlashTrace 长上下文耗时实验（exp1）

自包含脚本：`exp/exp1/run_time_curve.py`  
用途：在单个 RULER 样本上，测量不同上下文长度下各归因方法的 wall-clock 时间与 GPU 峰值显存，供论文中的线性增长表格使用。

## 方法覆盖
- `IG`（20 步）
- `attention_I_G`（注意力 * IG）
- `attnlrp`（单次反传的 LRP 版本）
- `perturbation_all`（log-loss ablation）
- `perturbation_CLP`（KL 版）
- `perturbation_REAGENT`（MLM 替换，LED/4096 上限，超过则可能失败）
- `ifr_all_positions`（IFR one-by-one baseline，`sink_chunk_tokens=1` 固定）
- `ifr_multi_hop`（FlashTrace，多跳+chunk 支持）
- `ifr_multi_hop_both`（FT-IFR both：stop_words + in_all_gen，多跳+chunk 支持）

## 运行示例
```bash
# 默认 input 长度 1024,4096,8192，output 长度 32,256,512；每格 3 次
python exp/exp1/run_time_curve.py \
  --model qwen-8B \
  --model_path /opt/share/models/Qwen/Qwen3-8B/ \
  --cuda 0,5,7 \
  --attr_funcs IG,perturbation_all,attention_I_G,perturbation_REAGENT,ifr_all_positions,perturbation_CLP,ifr_multi_hop,ifr_multi_hop_both,attnlrp \
  --input_lengths 10 \
  --output_lengths 10,100,500,1000,2000,5000,10000 \
  --repeats 3 \
  --chunk_tokens 128 \
  --sink_chunk_tokens 32 \
  --catch_oom \
  --ruler_file data/ruler_multihop/8192/vt_h10_c1/validation.jsonl
```

输出：
- `exp/exp1/out/time_curve_runs.jsonl`：每次运行的原始记录（attr、目标 input/output/total、实际长度、time、peak_mem、status）。
- `exp/exp1/out/time_curve_summary.csv`：按方法 + 目标 input/output 汇总的均值/方差（同时写出 total=input+output）。

## 注意事项
- `--input_lengths` 控制 prompt（user prompt）长度，`--output_lengths` 控制 output（sink）长度；每个格子的 total = input + output。
- 兼容：仍支持 `--total_lengths/--lengths`（deprecated），表示 prompt+output 总长度；prompt 长度按两者差值生成。
- `--target_text` 作为基底被重复拼接以满足目标 output 长度，仅用于控制长度，不在乎语义。
- `--catch_oom/--no-catch-oom` 用于选择是把 OOM 记为 status 继续，还是直接抛错中止。
- 多卡：`--cuda 0,1` 会在脚本启动前设置 `CUDA_VISIBLE_DEVICES` 并用 `device_map=balanced` 分片加载；单卡指定 `--cuda 0`。
- 超出模型上下文 (`config.max_position_embeddings`) 会标记 `skipped_model_ctx`（按实际喂给模型的 formatted prompt + output(+eos) token 数检查）。
- `perturbation_REAGENT` 的 Longformer 仅支持 4096 tokens，超过可能返回 OOM 或 runtime_error。
- IFR multi-hop 提供 `--chunk_tokens/--sink_chunk_tokens` 以在超长上下文上强制分块，显存会下降但时间略升；`ifr_all_positions` 分支固定 `sink_chunk_tokens=1`。
