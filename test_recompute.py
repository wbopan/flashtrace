"""Smoke test: verify attention recomputation matches output_attentions=True."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# --- 1. Create a tiny Qwen2-style model from scratch ---
print("Creating tiny Qwen2 model...")
config = AutoConfig.for_model(
    "qwen2",
    vocab_size=1000,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=128,
    use_sliding_window=False,
    attn_implementation="eager",
)
model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
model.eval()
print(f"  layers={config.num_hidden_layers}, heads={config.num_attention_heads}, "
      f"kv_heads={config.num_key_value_heads}, d_model={config.hidden_size}")

# --- 2. Create dummy input ---
seq_len = 16
input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
attention_mask = torch.ones_like(input_ids)

# --- 3. Get ground truth attention from model ---
print("\nRunning forward with output_attentions=True...")
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        use_cache=False,
        return_dict=True,
    )
gt_attentions = outputs.attentions  # tuple of [1, n_heads_q, S, S] per layer
print(f"  Got {len(gt_attentions)} layers, shape={gt_attentions[0].shape}")

# --- 4. Test recompute_layer_attention ---
print("\nTesting recompute_layer_attention...")
from ifr_core import (
    extract_model_metadata,
    build_weight_pack,
    recompute_layer_attention,
    IFRParameters,
    attach_hooks,
)

metadata = extract_model_metadata(model)
print(f"  rotary_emb found: {metadata.rotary_emb is not None}")

model_dtype = next(model.parameters()).dtype
weight_pack = build_weight_pack(metadata, model_dtype)

# Capture intermediate activations
cache, hooks = attach_hooks(metadata.layers, model_dtype)
with torch.no_grad():
    _ = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=False,
        use_cache=False,
        return_dict=True,
    )
for h in hooks:
    h.remove()

params = IFRParameters(
    n_layers=metadata.n_layers,
    n_heads_q=metadata.n_heads_q,
    n_kv_heads=metadata.n_kv_heads,
    head_dim=metadata.head_dim,
    group_size=metadata.group_size,
    d_model=metadata.d_model,
    sequence_length=seq_len,
    model_dtype=model_dtype,
    chunk_tokens=128,
    sink_chunk_tokens=32,
)

max_diff_all = 0.0
for li in range(metadata.n_layers):
    x_prev = cache["pre_attn_resid"][li][0]  # [S, d_model]
    recomputed = recompute_layer_attention(x_prev, weight_pack[li], metadata.rotary_emb, params)
    ground_truth = gt_attentions[li][0]  # [n_heads_q, S, S]

    diff = (recomputed - ground_truth).abs().max().item()
    max_diff_all = max(max_diff_all, diff)
    print(f"  Layer {li}: max_diff={diff:.2e}, "
          f"recomputed_sum={recomputed.sum():.4f}, gt_sum={ground_truth.sum():.4f}")

print(f"\n  Overall max diff across all layers: {max_diff_all:.2e}")
if max_diff_all < 1e-5:
    print("  PASS: Recomputed attention matches ground truth!")
else:
    print("  FAIL: Significant difference detected!")

# --- 5. Test full IFR pipeline in both modes ---
print("\n\nTesting full IFR pipeline (both modes)...")
from ifr_core import compute_ifr_for_all_positions

# Mode A: with stored attentions
result_a = compute_ifr_for_all_positions(
    cache=cache,
    attentions=gt_attentions,
    weight_pack=weight_pack,
    params=params,
    sink_range=(4, seq_len - 1),
)

# Mode B: with recomputed attentions
result_b = compute_ifr_for_all_positions(
    cache=cache,
    attentions=None,
    weight_pack=weight_pack,
    params=params,
    sink_range=(4, seq_len - 1),
    rotary_emb=metadata.rotary_emb,
)

token_diff = (result_a.token_importance_matrix - result_b.token_importance_matrix).abs().max().item()
head_diff = (result_a.head_importance_matrix - result_b.head_importance_matrix).abs().max().item()
print(f"  token_importance max_diff: {token_diff:.2e}")
print(f"  head_importance max_diff: {head_diff:.2e}")

if token_diff < 1e-4 and head_diff < 1e-4:
    print("  PASS: Full IFR pipeline produces matching results!")
else:
    print("  FAIL: IFR results differ between modes!")

print("\nAll tests complete.")
