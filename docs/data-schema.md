# Data schema

The local dashboard imports one UTF-8 CSV through the file picker. One file is one
property.

## Columns

| column | required | type | notes |
| --- | --- | --- | --- |
| `review_id` | yes | text | Unique within the file. |
| `date` | yes | ISO date | `YYYY-MM-DD`. Empty dates are kept but ignored by the period. |
| `rating` | yes | number | 1 to 5. Empty ratings become `unknown` in the rating bucket. |
| `text` | no | text | Review body. Empty means rating-only. |

Extra columns are ignored. A row with a duplicate `review_id` is rejected. Exact
duplicate texts are removed; near-duplicates are removed at a high token-overlap
guard. Text shorter than 30 normalised characters is not deduplicated.

## Example row

This row is synthetic, not a real review:

```csv
review_id,date,rating,text
r-0001,2025-06-14,4,"The room was clean and the staff were helpful. Breakfast was fresh, and the pool was quiet in the morning."
```

## Importing a batch

1. Start the local app with `run.bat`.
2. In the browser, choose or drop a CSV in the import panel.
3. The server writes the file under `data/imports/`, replaces the current
   single-property dataset, and runs normalize, classify, sentiment, facets,
   aggregate and the dashboard data build.
4. The page reloads the overview when the import finishes.

The upload limit is 20 MB. The import route is local-only (`127.0.0.1`).

`data/` is not tracked by git. The first import creates it with the dataset,
`stats.json`, the dashboard data and a copy of the uploaded CSV.

Before the dataset is cleared, the server checks the header and every row:
missing columns, empty `review_id`, duplicate ids, non-ISO dates, and ratings
outside 1 to 5 are rejected with the row number. The page switches back to the
import panel and shows the error. If the pipeline itself fails after that check,
the current dataset has already been cleared; fix the input and import again.

## Sample data

```bat
python scripts\make_sample_data.py
```

This writes `data/sample/reviews.csv`: 400 synthetic rows for one property, in
English. The generator is deterministic and the text is built from templates, not
copied from any real review.

## Showcase data

The static page in `docs/` does not import CSV. It reads a committed cache built
from a local CSV under `data/showcase/`:

```bat
python scripts\build_showcase.py
```

That CSV has the same columns and is git-ignored. The cache in
`docs/data/showcase.json` is the only showcase data in the repository. Rerun the
script after changing the text to refresh it.
