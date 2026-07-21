# Independent Wiki-VISA native n=10 audit

I read the dataset, model, and generation-evaluation artifacts separately and
visually inspected all ten native screenshots, including their evidence
regions.  The audit JSONL contains judgments only; it does not duplicate the
dataset records or model-generated text.

## Main result

- Dataset rows: 10; preserved model records: 8.  Rows `0882` and `1531` have
  no auditable model record.
- Among the eight preserved whole outputs, six are plainly correct and two are
  semantically acceptable scope mismatches (`0384`, `2171`).  I found no
  answer-level hallucination.
- The automatic normalizer marks four of these eight outputs wrong.  All four
  are semantic false negatives:
  - `0384`: the evidence defines joey as an infant marsupial and gives Koala as
    an example; the model chooses the valid general class.
  - `1762`: Ned Leeds is a correct, more specific form of Ned.
  - `2114`: a full-sentence paraphrase preserves the reference meaning.
  - `2171`: all three extra vocalists are explicitly supported, although this
    is wider than the one-name reference.
- Manual image-dependence judgments agree with the blur-log-probability gate on
  all eight preserved records: dependent for `1449`, `2114`, `2171`, `2870`;
  not dependent for `0052`, `0384`, `1762`, `2759`.
- Only `0384` and `2114` contain genuinely non-trivial reasoning.  The other
  six preserved THINKING spans are mostly verbose direct retrieval.  `0384`
  mildly loops while resolving class versus example.  `2759` misreads the
  visible working title **Hey Jules** as **Hey Jude** inside THINKING, although
  its final output remains correct.

## Two failed rows

- `0882` is both a dataset and generation failure.  Jacksonville has never
  reached a Super Bowl; its infobox shows zero conference championships.
  `2017` is a division-title/playoff season, not a Super Bowl appearance.  The
  model hit the 768-token limit without producing `</think>`, and the retained
  tail indicates repeated AFC-Championship-versus-Super-Bowl reconsideration.
- `1531` has a visually supported reference: the passage identifies Marla as
  Trish's teenage daughter and names Kat Dennings.  But generation was rejected
  because generated token IDs did not equal decode/re-encoded teacher-forced
  IDs, so neither its whole OUTPUT nor THINKING can be independently audited.

## Pilot implication

Only `1449` and `2870` satisfy the existing strict automated gate unchanged,
and both are simple passage lookup rather than long/open reasoning.  Semantic
whole-output judging would additionally recover `2114` and `2171` as correct,
image-dependent cases, but `2171` should be labeled wider-than-reference.
This native n=10 therefore exposes a major exact-normalization problem and does
not yet demonstrate a strong long/open visual-reasoning pilot.

Visual inspection aids created under `/tmp`:

- `/tmp/wiki_native_n10_full_sheet.jpg`
- `/tmp/wiki_native_n10_evidence_crops.jpg`
