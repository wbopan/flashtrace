# FlashTrace 长上下文耗时实验（exp1）

自包含脚本：`exp/exp1/run_time_curve.py`  
用途：在单个 RULER 样本上，测量不同上下文长度下各归因方法的 wall-clock 时间与 GPU 峰值显存，供论文中的线性增长表格使用。

## 方法覆盖
- `IG`（20 步）
- `attention_I_G`（注意力 * IG）
- `perturbation_all`（log-loss ablation）
- `perturbation_CLP`（KL 版）
- `perturbation_REAGENT`（MLM 替换，LED/4096 上限，超过则可能失败）
- `ifr_all_positions`（IFR one-by-one baseline）
- `ifr_multi_hop`（FlashTrace，多跳+chunk 支持）

## 运行示例
```bash
# 默认长度 0,100,500,1000,5000,100000；每格 3 次
python exp/exp1/run_time_curve.py \
  --model qwen-8B \
  --model_path /opt/share/models/Qwen/Qwen3-8B/ \
  --cuda 0 \
  --attr_funcs IG,attention_I_G,perturbation_all,perturbation_CLP,perturbation_REAGENT,ifr_all_positions,ifr_multi_hop \
  --lengths 0,100,500,1000,5000,100000 \
  --repeats 3 \
  --chunk_tokens 128 \
  --sink_chunk_tokens 32 \
  --ruler_file data/ruler_multihop/8192/vt_h10_c1/validation.jsonl
```

输出：
- `exp/exp1/out/time_curve_runs.jsonl`：每次运行的原始记录（attr、长度、time、mem、status）。
- `exp/exp1/out/time_curve_summary.csv`：按方法+长度汇总的均值/方差。

## 注意事项
- `--target_text` 固定为短句，避免生成随机 CoT；只关心时间/显存。
- 超出模型上下文 (`config.max_position_embeddings`) 会标记 `skipped_model_ctx`。
- `perturbation_REAGENT` 的 Longformer 仅支持 4096 tokens，超过可能返回 OOM 或 runtime_error。
- IFR 提供 `--chunk_tokens/--sink_chunk_tokens` 以在超长上下文上强制分块，显存会下降但时间略升。
