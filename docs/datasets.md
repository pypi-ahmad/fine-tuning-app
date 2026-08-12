# Datasets

## Sources

Hugging Face datasets use a dataset ID, revision (default `main`), and split (default
`train`). Authentication for gated or private datasets comes from `HF_TOKEN`.

Local uploads support `.json`, `.jsonl`, `.csv`, and `.parquet`. The UI previews up to
100 rows. The full file is copied into the job directory before the background worker
starts.

## Column mappings

### Text

Select one column containing complete training examples. The default column is `text`.

### Prompt / response

Select separate prompt and response columns (defaults: `prompt` and `response`). The
worker combines them using the selected template:

- `alpaca`: instruction and response headings.
- `chatml`: user and assistant ChatML markers.
- `plain` or `tokenizer`: prompt followed by response as plain text.

### Messages

Select a column containing a list of `{role, content}` message objects. With the
`tokenizer` template, the model tokenizer's chat template is used when available.
`chatml` adds ChatML markers; other selections render `Role: content` lines.

## Validation split

The validation fraction can be 0.05 through 0.40 and defaults to 0.10. The default
shuffle seed is 42. A dataset must contain at least two rows, and every configured
column must exist.

Dataset preview validates file parsing, but full column and row validation occurs in
the training worker.
