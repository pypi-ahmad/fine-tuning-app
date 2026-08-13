# Datasets

## Sources

The Dataset page starts with one source. **Add dataset** appends another; **Remove**
deletes a source when more than one remains. Hub and local uploads can be mixed in the
same run.

Hugging Face datasets accept either a dataset ID such as `MatrAIx2026/MatrAIx_Persona_1M`
or its `https://huggingface.co/datasets/...` repository URL. Revision defaults to `main`
and split defaults to `train`. **Validate & download dataset** checks the reference,
reports its files and bytes, and downloads it into the standard Hugging Face cache.
Authentication for gated or private datasets comes from `HF_TOKEN` (preferred) or the
legacy `HUGGING_FACE_HUB_TOKEN` alias.

Tree, viewer, blob, and resolve URLs are normalized to the containing repository. The
Revision field remains authoritative; a pasted file URL does not select only that file.

Local uploads support `.json`, `.jsonl`, `.csv`, and `.parquet`. The UI previews up to
100 rows. Each upload is copied into `inputs/dataset-<n>/` in the job directory before
the background worker starts.

## Combining sources

Each source has its own split, mapping, and template. The worker loads and normalizes
every source, concatenates the results, then shuffles and splits once using the global
validation fraction and seed. Rows are not weighted or deduplicated. Concatenation
fails if the prepared sources do not share compatible columns and value types.

Match the data shape to the training approach. Preference, KTO, reward, PPO, ORPO,
SimPO, and GRPO jobs use the recipe's required columns rather than converting rows to
a single `text` field.

## Column mappings

### Text

Select one column containing complete training examples. The default column is `text`.
Used for supervised fine-tuning and continued pretraining.

### Prompt / response

Select separate prompt and response columns (defaults: `prompt` and `response`). The
worker combines them into `text` using the selected template:

- `alpaca`: instruction and response headings.
- `chatml`: user and assistant ChatML markers.
- `plain` or `tokenizer`: prompt followed by response as plain text.

### Messages

Select a column containing a list of `{role, content}` message objects. With the
`tokenizer` template, the model tokenizer's chat template is used when available.
`chatml` adds ChatML markers; other selections render `Role: content` lines.

### Preference

Requires `prompt`, `chosen`, and `rejected`. Use this shape for DPO, ORPO, SimPO, and
reward modeling (`chosen` and `rejected` only for reward).

### KTO

Requires `prompt`, `completion`, and a boolean `label` column.

### GRPO

Requires a `prompt` column. PPO uses the same `prompt` contract.

## Validation split

**Global validation fraction** can be 0.05 through 0.40 and defaults to 0.10. **Global
split seed** defaults to 42. Each source must contain at least two rows, and every
column required by that source's mapping must exist.

Dataset preview validates file parsing, but full column, row, and combine validation
occurs in the training worker.
