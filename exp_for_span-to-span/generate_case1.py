import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-8B"

CASE_PROMPT = """About 3 years ago or so I was skinny, but I was still ugly. I really do want to change that but I've tried 3 times. Now can I try but don't stop. At school I get bullied about my weight and my ugliness and I have been bullied my whole life that I believe them. How can I stop thinking about them and don't let it get in my head? My parents said I don't weight that much but, they do think I am fat but, they tell me I am not. I told my parents that I need to go to a therapist, but they think I am fine and I think they don't want to deal with it."""

# Encourage a reasonably sized reasoning trace for our span-level case study.
messages = [
    {
        "role": "system",
        "content": (
            "Respond helpfully and safely. Before the final response, reason in a "
            "concise sequence of complete sentences before giving the final response."
        ),
    },
    {"role": "user", "content": CASE_PROMPT},
]

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)
model.eval()

# Qwen3 native thinking mode.
formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)

model_inputs = tokenizer(
    formatted_prompt,
    return_tensors="pt",
).to(model.device)

input_len = model_inputs.input_ids.shape[1]

torch.manual_seed(0)

with torch.no_grad():
    generated = model.generate(
        **model_inputs,
        max_new_tokens=1200,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
    )

# Only the newly generated assistant tokens.
output_ids = generated[0, input_len:].tolist()

# Qwen3 uses </think> to separate reasoning from final response.
think_end_id = tokenizer.convert_tokens_to_ids("</think>")

try:
    think_end_pos = len(output_ids) - 1 - output_ids[::-1].index(think_end_id)

    reasoning_ids = output_ids[:think_end_pos]
    final_ids = output_ids[think_end_pos + 1:]
except ValueError:
    raise RuntimeError("No </think> token found in generation.")

reasoning_text = tokenizer.decode(
    reasoning_ids,
    skip_special_tokens=True,
).strip()

final_text = tokenizer.decode(
    final_ids,
    skip_special_tokens=True,
).strip()

# Keep BOTH text and token IDs.
# Token IDs guarantee that we retain the exact frozen generation.
result = {
    "model": MODEL_NAME,
    "prompt": CASE_PROMPT,
    "reasoning_text": reasoning_text,
    "final_text": final_text,
    "reasoning_token_ids": reasoning_ids,
    "final_token_ids": final_ids,
    "full_generation_token_ids": output_ids,
    "think_end_position": think_end_pos,
}

with open(
    "exp_for_span-to-span/results/case1_generation.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print("\n================ REASONING ================\n")
print(reasoning_text)

print("\n================ FINAL OUTPUT =============\n")
print(final_text)

print("\nSaved to exp_for_span-to-span/results/case1_generation.json")