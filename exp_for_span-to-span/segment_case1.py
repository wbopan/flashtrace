import json
from transformers import AutoTokenizer

PATH = "exp_for_span-to-span/results/case1_generation.json"

with open(PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(data["model"])

reasoning_ids = data["reasoning_token_ids"]

# Skip the opening <think> token(s)
start = 0
while start < len(reasoning_ids):
    piece = tokenizer.decode([reasoning_ids[start]])
    if "<think>" in piece or piece.strip() == "":
        start += 1
    else:
        break

spans = []
sent_start = start

for idx in range(start, len(reasoning_ids)):
    piece = tokenizer.decode([reasoning_ids[idx]])

    # Natural-language boundary for the prototype: "."
    if "." in piece:
        sent_end = idx

        text = tokenizer.decode(
            reasoning_ids[sent_start:sent_end + 1],
            skip_special_tokens=True,
        ).strip()

        if text:
            spans.append({
                "id": f"S{len(spans) + 1}",
                "start": sent_start,
                "end": sent_end,
                "text": text,
            })

        sent_start = idx + 1

# Handle any remaining text after the last period
if sent_start < len(reasoning_ids):
    text = tokenizer.decode(
        reasoning_ids[sent_start:],
        skip_special_tokens=True,
    ).strip()

    if text:
        spans.append({
            "id": f"S{len(spans) + 1}",
            "start": sent_start,
            "end": len(reasoning_ids) - 1,
            "text": text,
        })

for s in spans:
    print(f"{s['id']} [{s['start']}:{s['end']}]  {s['text']}")

data["reasoning_spans"] = spans

with open(PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)