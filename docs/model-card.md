# Model card: review category classifier

## What it is

A DeBERTa-v3-base sequence classifier with seven sigmoid outputs. It labels a
review sentence with every category present. The
shipped checkpoint is a single seed, `cat_v5c_s2`.

| id | label | scored | shown in dashboard |
| --- | --- | --- | --- |
| 0 | Room & Property | yes | yes |
| 1 | Food & Drink | yes | yes |
| 2 | Staff & Service | yes | yes |
| 3 | Location | no | no |
| 4 | Pool & Beach | yes | yes |
| 5 | Spa & Wellness | yes | yes |
| 6 | Entertainment & Animation | yes | yes |

`Location` stays in the checkpoint because removing an output would change the
label order. It is treated as an `Other` catch-all, excluded from the score, and
excluded from aggregation and the dashboard.

## Provenance

The classifier is a DeBERTa-v3-base fine-tuned on hand-labelled public reviews from hotels in the same region, and scored on a held-out property that was absent from training. The training corpus and the evaluation gold set are not released. The reported score is therefore not independently reproducible from this repository.

## Measured score

Protocol: one example per (review id, snippet), a neutral mention counts as
category-present, six actionable categories, `Location` excluded, per-label
thresholds from the checkpoint's own `thresholds.json`.

| seed | micro P | micro R | micro F1 | category-macro P | category-macro R | category-macro F1 |
| --- | --- | --- | --- | --- | --- | --- |
| `cat_v5c` | 0.864 | 0.908 | 0.885 | 0.857 | 0.909 | 0.882 |
| `cat_v5c_s1` | 0.838 | 0.897 | 0.867 | 0.883 | 0.860 | 0.865 |
| `cat_v5c_s2` | 0.867 | 0.900 | 0.883 | 0.889 | 0.894 | 0.890 |

Winner: `cat_v5c_s2`, category-macro F1 **0.890**. Spread across the three seeds:
**0.025**.

## Per-label thresholds

Thresholds are applied to each sigmoid independently. They are not a softmax and
do not sum to one.

| label | threshold |
| --- | --- |
| Room & Property | 0.35 |
| Food & Drink | 0.65 |
| Staff & Service | 0.50 |
| Location | 0.70 |
| Pool & Beach | 0.40 |
| Spa & Wellness | 0.55 |
| Entertainment & Animation | 0.70 |

The shipped file is `config/thresholds.json`. A local refit writes
`models/cascade/thresholds.json`, which takes precedence. The older
`config/cascade/thresholds.json` is kept as a runtime fallback.

## Files

`config.json`, `model.safetensors`, `tokenizer.json`, `spm.model`,
`special_tokens_map.json`, `added_tokens.json`, `thresholds.json`.

The float32 checkpoint is 737,734,660 bytes (703.6 MiB). The shipped fp16 file is
368,879,606 bytes (351.8 MiB). Weights are not tracked in git; `models/` is
ignored.

## Loading

`catmodel.py` accepts a local directory or a Hugging Face id:

```bat
set REVIEWS_MODEL=hf:your-account/deberta-review-categories
python app\pipeline\cascade_extract.py
```

or

```bat
python app\pipeline\cascade_extract.py --model-path models\cat
```

The default is `models/cat` when it exists, otherwise the `hf:` placeholder in
`catmodel.DEFAULT_MODEL`.

## Sentiment probe

The sentiment stage is separate: `intfloat/multilingual-e5-large-instruct`
embeddings plus a logistic regression saved as `config/cascade/probe.joblib`.
Scores within 0.10 of 0.5 are labelled neutral. The probe is fitted offline by
`training/fit_probe.py`; the service only loads it.

## Limits

One hospitality domain, English only, sentence-level classification. The
`General Experience` facet is a vague catch-all, shown separately in the local
dashboard and omitted from the showcase. The model was not trained on other
domains. Training-data rights remain with the review platforms and the authors of
the reviews.
