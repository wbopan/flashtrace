import json
import numpy as np
import torch

from flashtrace import load_model_and_tokenizer
from flashtrace.attribution import LLMIFRAttribution
from flashtrace.core import compute_ifr_span_to_span_aggregate


JSON_PATH = "exp_for_span-to-span/results/case1_generation.json"
OUTPUT_PATH = "exp_for_span-to-span/results/case1_span_matrix.json"

# Must be EXACTLY the same system prompt used during generation.
SYSTEM_PROMPT = (
    "Respond helpfully and safely. Before the final response, reason in a "
    "concise sequence of complete sentences. "
    "Use enough reasoning steps to make the reasoning process explicit, "
    "but avoid unnecessary verbosity."
)


# =========================================================
# 1. Load frozen case
# =========================================================

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

model_name = data["model"]

print("Loading model...")
model, tokenizer = load_model_and_tokenizer(model_name)

engine = LLMIFRAttribution(
    model,
    tokenizer,
    chunk_tokens=128,
    sink_chunk_tokens=32,
    show_progress=False,
    recompute_attention=False,
)


# =========================================================
# 2. Reconstruct exact prompt + locate user query Q
# =========================================================

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": data["prompt"]},
]

formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)

# Keep offsets so we can identify the user query token span.
prompt_encoding = tokenizer(
    formatted_prompt,
    return_tensors="pt",
    return_offsets_mapping=True,
)

offsets = prompt_encoding.pop("offset_mapping")[0].tolist()

prompt_inputs = {
    k: v.to(engine.device) if torch.is_tensor(v) else v
    for k, v in prompt_encoding.items()
}

prompt_ids = prompt_inputs["input_ids"]
prompt_len = int(prompt_ids.shape[1])


# Locate query characters inside the formatted chat prompt.
query_char_start = formatted_prompt.index(data["prompt"])
query_char_end = query_char_start + len(data["prompt"])

query_token_indices = [
    i
    for i, (start, end) in enumerate(offsets)
    if end > query_char_start and start < query_char_end
]

query_abs = (
    query_token_indices[0],
    query_token_indices[-1],
)

print(f"Prompt length: {prompt_len}")
print(f"QUERY Q: absolute [{query_abs[0]}:{query_abs[1]}]")
print(
    "Decoded Q:",
    tokenizer.decode(
        prompt_ids[0, query_abs[0]:query_abs[1] + 1],
        skip_special_tokens=True,
    ),
)


# =========================================================
# 3. Reuse exact frozen generation
# =========================================================

generation_ids = torch.tensor(
    [data["full_generation_token_ids"]],
    dtype=prompt_ids.dtype,
    device=engine.device,
)

gen_len = int(generation_ids.shape[1])

input_ids_all = torch.cat(
    [prompt_ids, generation_ids],
    dim=1,
)

attention_mask = torch.ones_like(input_ids_all)

total_len = int(input_ids_all.shape[1])

engine._prompt_model_inputs = dict(prompt_inputs)
engine.prompt_ids = prompt_ids
engine.generation_ids = generation_ids


# =========================================================
# 4. Capture model internals once
# =========================================================

print("\nCapturing model internals...")

cache, attentions, metadata, weight_pack = engine._capture_model_state(
    input_ids_all,
    attention_mask,
    recompute_attention=False,
)

params = engine._build_ifr_params(
    metadata,
    total_len,
)

print("Model state captured.")


# =========================================================
# 5. Reasoning sentence spans → absolute token indices
# =========================================================

reasoning_spans = data["reasoning_spans"]
reasoning_abs = []

for span in reasoning_spans:
    start_abs = prompt_len + int(span["start"])
    end_abs = prompt_len + int(span["end"])

    reasoning_abs.append((start_abs, end_abs))

    print(
        f"{span['id']}: "
        f"local [{span['start']}:{span['end']}] "
        f"-> absolute [{start_abs}:{end_abs}]"
    )


# =========================================================
# 6. Locate final output span O
# =========================================================

output_start_local = int(data["think_end_position"]) + 1
output_end_local = gen_len - 1

eos_id = tokenizer.eos_token_id
pad_id = tokenizer.pad_token_id


def ignorable_token(token_id):
    if eos_id is not None and token_id == eos_id:
        return True

    if pad_id is not None and token_id == pad_id:
        return True

    text = tokenizer.decode(
        [token_id],
        skip_special_tokens=True,
    )

    return text.strip() == ""


# Remove whitespace after </think>.
while (
    output_start_local <= output_end_local
    and ignorable_token(int(generation_ids[0, output_start_local]))
):
    output_start_local += 1


# Remove trailing whitespace / EOS.
while (
    output_end_local >= output_start_local
    and ignorable_token(int(generation_ids[0, output_end_local]))
):
    output_end_local -= 1


output_abs = (
    prompt_len + output_start_local,
    prompt_len + output_end_local,
)

print(
    f"\nOUTPUT O: local [{output_start_local}:{output_end_local}] "
    f"-> absolute [{output_abs[0]}:{output_abs[1]}]"
)


# =========================================================
# 7. Create full span attribution matrix
#
# Node order:
#
# Q, S1, S2, ..., S24, O
#
# rows    = TARGET
# columns = SOURCE
#
# matrix[target, source] = source → target attribution
# =========================================================

num_reasoning = len(reasoning_spans)

labels = (
    ["Q"]
    + [span["id"] for span in reasoning_spans]
    + ["O"]
)

num_nodes = num_reasoning + 2
output_idx = num_nodes - 1

matrix_raw = np.full(
    (num_nodes, num_nodes),
    np.nan,
    dtype=np.float32,
)


# =========================================================
# 8. Q + earlier reasoning → each reasoning sentence
#
# S1: Q → S1
# S2: Q,S1 → S2
# S3: Q,S1,S2 → S3
# ...
# =========================================================

for target_ridx in range(num_reasoning):

    target_start, target_end = reasoning_abs[target_ridx]

    # Query plus every earlier reasoning sentence.
    source_spans = (
        [query_abs]
        + reasoning_abs[:target_ridx]
    )

    # +1 because matrix index 0 is Q.
    target_matrix_idx = target_ridx + 1

    print(
        f"\nComputing "
        f"{labels[:target_matrix_idx]} "
        f"-> {labels[target_matrix_idx]}"
    )

    result = compute_ifr_span_to_span_aggregate(
        sink_start=target_start,
        sink_end=target_end,
        source_spans=source_spans,
        cache=cache,
        attentions=attentions,
        weight_pack=weight_pack,
        params=params,
        renorm_threshold=0.0,
        sink_weights=None,
        rotary_emb=metadata.rotary_emb,
    )

    scores = (
        result.span_importance_total
        .detach()
        .cpu()
        .numpy()
    )

    matrix_raw[
        target_matrix_idx,
        :target_matrix_idx
    ] = scores


# =========================================================
# 9. Q + all reasoning spans → Output O
# =========================================================

print("\nComputing Q, S1...S24 -> O")

output_source_spans = (
    [query_abs]
    + reasoning_abs
)

result_output = compute_ifr_span_to_span_aggregate(
    sink_start=output_abs[0],
    sink_end=output_abs[1],
    source_spans=output_source_spans,
    cache=cache,
    attentions=attentions,
    weight_pack=weight_pack,
    params=params,
    renorm_threshold=0.0,
    sink_weights=None,
    rotary_emb=metadata.rotary_emb,
)

output_scores = (
    result_output.span_importance_total
    .detach()
    .cpu()
    .numpy()
)

matrix_raw[
    output_idx,
    :output_idx
] = output_scores


# =========================================================
# 10. Row-normalized matrix
# =========================================================

matrix_normalized = matrix_raw.copy()

for row in range(num_nodes):

    valid = np.isfinite(matrix_normalized[row])

    if not valid.any():
        continue

    row_sum = matrix_normalized[row, valid].sum()

    if row_sum > 0:
        matrix_normalized[row, valid] /= row_sum


# =========================================================
# 11. Save
# =========================================================

def json_matrix(matrix):
    return [
        [
            None if not np.isfinite(value) else float(value)
            for value in row
        ]
        for row in matrix
    ]


result_json = {
    "labels": labels,

    "orientation": "rows=target, columns=source",

    "raw_matrix": json_matrix(matrix_raw),
    "normalized_matrix": json_matrix(matrix_normalized),

    "query_span_absolute": {
        "start": int(query_abs[0]),
        "end": int(query_abs[1]),
        "text": data["prompt"],
    },

    "reasoning_spans": reasoning_spans,

    "reasoning_absolute_spans": [
        {
            "id": reasoning_spans[i]["id"],
            "start": int(start),
            "end": int(end),
        }
        for i, (start, end) in enumerate(reasoning_abs)
    ],

    "output_span_local": {
        "start": int(output_start_local),
        "end": int(output_end_local),
    },

    "output_span_absolute": {
        "start": int(output_abs[0]),
        "end": int(output_abs[1]),
    },
}


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        result_json,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =========================================================
# 12. Quick sanity check
# =========================================================

source_texts = (
    [data["prompt"]]
    + [span["text"] for span in reasoning_spans]
)

source_labels = labels[:-1]

order = np.argsort(output_scores)[::-1]

print("\n========== TOP SPANS -> OUTPUT ==========")

for idx in order[:10]:
    print(
        f"{source_labels[idx]} "
        f"{output_scores[idx]:.6f} | "
        f"{source_texts[idx]}"
    )


print(f"\nMatrix shape: {matrix_raw.shape}")
print(f"Saved: {OUTPUT_PATH}")