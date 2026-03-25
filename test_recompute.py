"""End-to-end test: verify recompute_attention mode produces identical IFR results
through the full LLMIFRAttribution pipeline."""

import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

# --- 1. Create a tiny Qwen2-style model ---
print("1. Creating tiny Qwen2 model...")
config = AutoConfig.for_model(
    "qwen2",
    vocab_size=500,
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
print(f"   layers={config.num_hidden_layers}, heads={config.num_attention_heads}, "
      f"kv_heads={config.num_key_value_heads}, d_model={config.hidden_size}")

# --- 2. Create a minimal tokenizer with chat template ---
print("2. Creating minimal tokenizer...")
tok_backend = Tokenizer(models.WordLevel(vocab={f"t{i}": i for i in range(500)}, unk_token="t0"))
tok_backend.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tok_backend,
    eos_token="t1",
    pad_token="t2",
)
# Minimal chat template that just concatenates messages
tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"

# --- 3. Run full pipeline in BOTH modes ---
print("3. Running LLMIFRAttribution end-to-end...")
from llm_attr import LLMIFRAttribution

prompt = "t10 t20 t30 t40 t50"
target = "t60 t70 t80"

# Mode A: stored attentions (default)
attr_a = LLMIFRAttribution(model, tokenizer, recompute_attention=False)
result_a = attr_a.calculate_ifr_for_all_positions(prompt, target)
print(f"   Mode A (stored):    score_shape={result_a.attribution_matrix.shape}")

# Mode B: recomputed attentions
attr_b = LLMIFRAttribution(model, tokenizer, recompute_attention=True)
result_b = attr_b.calculate_ifr_for_all_positions(prompt, target)
print(f"   Mode B (recompute): score_shape={result_b.attribution_matrix.shape}")

# --- 4. Compare ---
print("\n4. Comparing results...")
diff = (result_a.attribution_matrix - result_b.attribution_matrix).abs()
max_diff = diff.max().item()
mean_diff = diff.mean().item()
print(f"   max_diff:  {max_diff:.2e}")
print(f"   mean_diff: {mean_diff:.2e}")

if max_diff < 1e-5:
    print("   PASS: End-to-end results match!")
else:
    print("   FAIL: Results differ!")
    # Print details for debugging
    print(f"   result_a:\n{result_a.attribution_matrix}")
    print(f"   result_b:\n{result_b.attribution_matrix}")

# --- 5. Also test span aggregate ---
print("\n5. Testing calculate_ifr_span...")
result_sa_a = attr_a.calculate_ifr_span(prompt, target)
result_sa_b = attr_b.calculate_ifr_span(prompt, target)
sa_diff = (result_sa_a.attribution_matrix - result_sa_b.attribution_matrix).abs().max().item()
print(f"   max_diff: {sa_diff:.2e}")
if sa_diff < 1e-5:
    print("   PASS")
else:
    print("   FAIL")

# --- 6. Also test multi-hop ---
print("\n6. Testing calculate_ifr_multi_hop...")
result_mh_a = attr_a.calculate_ifr_multi_hop(prompt, target, n_hops=2)
result_mh_b = attr_b.calculate_ifr_multi_hop(prompt, target, n_hops=2)
mh_diff = (result_mh_a.attribution_matrix - result_mh_b.attribution_matrix).abs().max().item()
print(f"   max_diff: {mh_diff:.2e}")
if mh_diff < 1e-5:
    print("   PASS")
else:
    print("   FAIL")

print("\nAll end-to-end tests complete.")
