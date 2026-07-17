# Qwen3-VL multimodal smoke results

Run date: 2026-07-16 (Asia/Hong_Kong)

Model: `Qwen/Qwen3-VL-8B-Instruct` at revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, bfloat16, one NVIDIA RTX PRO
5000 72GB Blackwell. Images were resized to at most 448 pixels on the long
side. The selected response was frozen before computing a 4x4 blur-region LOO
map over its mean token log-probability.

```bash
python -m evaluations.multimodal.prepare_data --samples 25
CUDA_VISIBLE_DEVICES=0 python -m evaluations.multimodal.run_smoke \
  --samples 5 \
  --correct-only \
  --candidate-limit 25 \
  --grid-size 4 \
  --max-image-side 448 \
  --output-dir data/multimodal_smoke_final
```

The selection scan retained the first five examples per dataset with VQA
consensus accuracy at least 0.6 and a reasoning sentence of at least three
words. VQA-X needed 5 candidates; A-OKVQA needed 11. The skipped responses are
recorded in `data/multimodal_smoke_final/attempts.jsonl`.

| Dataset | Question ID | Prediction | Majority answer | VQA acc. | Full-blur drop | Top cell | Top-quartile share |
| --- | --- | --- | --- | ---: | ---: | --- | ---: |
| VQA-X | 393271001 | yes | yes | 1.00 | 1.7808 | (1,1) | 0.853 |
| VQA-X | 393284000 | No | no | 1.00 | 2.2948 | (3,0) | 0.603 |
| VQA-X | 393338001 | tulips | tulips | 1.00 | 2.4642 | (0,1) | 0.670 |
| VQA-X | 524436004 | no | no | 1.00 | 3.0439 | (1,1) | 1.000 |
| VQA-X | 262531001 | skateboarding | skateboarding | 1.00 | 1.2366 | (2,2) | 0.896 |
| A-OKVQA | 22jbM6gDxdaMaunuzgrsBB | cigarette | cigarette | 1.00 | 1.2072 | (2,1) | 0.947 |
| A-OKVQA | 2C8riXpRLX3CyM5jDz23m7 | frosting | icing | 0.90 | 0.9516 | (1,2) | 0.568 |
| A-OKVQA | 2P5mVJc5a6DcCN9opV92FJ | surfing | surfing | 1.00 | 1.8290 | (1,1) | 0.883 |
| A-OKVQA | 2PGwvdFESLvwfFCwK5pbYu | jeans | jeans | 1.00 | 1.4070 | (3,3) | 0.895 |
| A-OKVQA | 2RN4dwhRZR3ZSKHtRnJdX3 | United States | united states | 1.00 | 1.3527 | (2,3) | 0.877 |

Aggregate checks:

- VQA-X mean VQA accuracy: 1.00; A-OKVQA: 0.98.
- All 10 responses contained both a reasoning sentence and a final answer.
- All 10 full-image blur perturbations reduced the frozen response
  log-probability.
- Mean top-quartile positive attribution share was 0.804 for VQA-X and 0.834
  for A-OKVQA.
- Mean generation/LOO times were 1.74/2.77 seconds for VQA-X and 1.30/3.16
  seconds for A-OKVQA. The ten selected samples took 44.83 seconds excluding
  model loading and rejected-candidate generation.

Manual overlay inspection found the strongest cells on the relevant objects
for the clearest positive questions: the flowers, skateboarder, surfer, and the
right-hand man's jeans. Negative questions such as "Is it raining?" produced
more diffuse maps; this is expected and is a reason to treat the smoke LOO map
as a pipeline sanity check rather than a paper-quality localization metric.
