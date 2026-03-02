# Vector DB Builder Module Technical Guide

Module file: `script/vdbbuilder.py`

## Purpose

`vdbbuilder.py` converts extracted filing text into item-level vector artifacts for peer search.

It performs:

1. scope-aware text loading (`all`, `heading`, `body`),
2. unitization and normalization,
3. embedding generation (local or OpenAI),
4. distinctiveness-weighted item pooling,
5. orthogonal common/specific decomposition,
6. artifact persistence (parquet, npz, optional FAISS indices).

## Scope behavior

- `--scope all`
  - Input: `*_item.json`
  - Reads `items[item_id].text_content` (fallback `html_content`).

- `--scope heading`
  - Input: `*_str.json`
  - Reads all nested `heading` values from `structures[item_id]`.

- `--scope body`
  - Input: `*_str.json`
  - Reads all nested `body` values from `structures[item_id]`.

All scopes are written into separate output namespaces:

```text
<out_dir>/scope=<scope>/...
```

## Year filter behavior

- `--year` / `--years` is optional.
- If omitted, builder scans and processes all years found in `--filing_dir`.
- If provided (e.g., `--year 2012 2013`), only those years are processed.

## End-to-end algorithm

For each `(firm_id, year, item_id)`:

1. Load item text by selected scope.
2. Canonicalize text:
   - normalize line endings,
   - collapse repeated spacing,
   - replace numeric patterns with tokens (`NUM`, `NEG_NUM`, `PCT`).
3. Chunk into overlapping token windows:
   - target length: `--chunk_tokens`
   - overlap: `--overlap_tokens`
   - drop chunks with tokens `< --min_unit_tokens`
4. Embed unit texts with selected backend.
5. Pool unit vectors using distinctiveness weighting:
   - centroid of units,
   - weight per unit proportional to distance from centroid + eps,
   - cap high weights by percentile (`--cap_weight_percentile`),
   - weighted average and L2 normalize.

For each `(item_id, year)` across firms:

6. Build pooled matrix `E` (L2 normalized rows).
7. Compute leave-one-out common direction per firm.
8. Orthogonally decompose each firm vector:
   - `alpha = dot(E_i, common_i)`
   - `residual_i = E_i - alpha * common_i`
9. If residual norm `< --residual_norm_floor`, mark residual missing.
10. Save pooled and residual artifacts.

For mathematical detail and interpretation of this decomposition in peer search,
see the **Orthogonal decomposition: intuition and math** section in `docs/peerfinder.md`.

## Embedding backends

### Local (`--embedder local`)

- deterministic hash-based pseudo-embedding (`dim=384`)
- no external API calls
- reproducible across runs for identical text

### OpenAI (`--embedder openai`)

- requires `OPENAI_API_KEY`
- requires `--embed_model` (e.g., `text-embedding-3-large`)
- batched calls (`--batch_size`)
- vectors are L2 normalized after API response

## Token counting

`TokenCounter` uses:

- `tiktoken` encoding for `--tokenizer-model` if available,
- fallback to whitespace tokenization otherwise.

This affects chunking and unit filtering thresholds.

## Output artifacts

All under:

```text
<out_dir>/scope=<scope>/
```

Typical directory structure:

```text
<out_dir>/
  scope=all|heading|body/
    units/
      units_2022.parquet
      units_2023.parquet
      units_2024.parquet
    item_vectors/
      item_vectors_2022.parquet
      item_vectors_2023.parquet
      item_vectors_2024.parquet
    vectors/
      pooled/
        item=1A/year=2024.npz
        item=7/year=2024.npz
      residual/
        item=1A/year=2024.npz
        item=7/year=2024.npz
    indices/                       # only when --build-faiss
      item=1A/year=2024/pooled.faiss
      item=1A/year=2024/residual.faiss
      item=1A/year=2024/pooled_ids.json
      item=1A/year=2024/residual_ids.json
```

### `units/units_<YEAR>.parquet`

Purpose:

- Unit-level audit table.
- Keeps the exact text chunks that were embedded.
- Main text source for `peerfinder --method gemini`.

Composition:

- one row per unit chunk.
- rows are keyed by `(firm_id, year, item_id, unit_id)`.

Key columns:

- `firm_id`, `year`, `item_id`, `unit_id`
- `unit_text`, `unit_tokens`
- `embedding_model`
- `embedding` (JSON string of vector values)
- `source_path`, `scope`

Example row (conceptual):

```text
firm_id=0000320193
year=2024
item_id=1A
unit_id=12
unit_tokens=243
scope=all
source_path=sec_filings/0000320193/2024/10-K/0000320193_2024_10-K_item.json
unit_text="...risk factors paragraph chunk..."
embedding="[0.014, -0.037, ...]"
```

### `item_vectors/item_vectors_<YEAR>.parquet`

Purpose:

- Item-level summary table after pooling/decomposition.
- Useful for diagnostics and analysis without loading NPZ files.

Composition:

- one row per `(firm_id, year, item_id)`.
- each row corresponds to multiple units from `units_<YEAR>.parquet`.

Key columns:

- `num_units`, `item_tokens`
- `pooled_embedding` (JSON vector)
- `w_max`, `w_mean` (distinctiveness-weight diagnostics)
- `common_loading`, `residual_norm`, `residual_embedding`
- `scope`

Example row (conceptual):

```text
firm_id=0000320193
year=2024
item_id=1A
num_units=38
item_tokens=9120
w_max=0.0821
w_mean=0.0317
common_loading=0.913
residual_norm=0.406
residual_embedding="[0.021, -0.005, ...]"   # null when residual below threshold
```

### `vectors/pooled/item=<ITEM>/year=<YEAR>.npz`

Purpose:

- Fast matrix for similarity search on pooled vectors.
- Primary retrieval artifact for cosine/common similarity.

Composition:

- `mat`: pooled matrix with shape `[N, D]`
  - `N`: firms available for this `(item, year)`
  - `D`: embedding dimension
- `ids`: aligned firm id array with shape `[N]`

Alignment rule:

- `mat[i]` always belongs to firm `ids[i]`.

### `vectors/residual/item=<ITEM>/year=<YEAR>.npz`

Purpose:

- Fast matrix for firm-specific (residual) similarity.
- Used by orthogonal method in peerfinder.

Composition:

- `mat`: residual matrix `[N, D]`
- `ids`: aligned firm ids `[N]`
- `mask`: residual-valid flag `[N]`
  - `1`: residual exists (norm above floor)
  - `0`: residual missing (row in `mat` is zero vector)

Interpretation:

- Peerfinder uses residual similarity only when both focal and peer have `mask=1`.
- Otherwise it falls back to pooled/common similarity.

### `indices/...` (optional FAISS)

Purpose:

- Prebuilt nearest-neighbor indexes for high-speed retrieval.

Generated when `--build-faiss`:

- `pooled.faiss` + `pooled_ids.json`
- `residual.faiss` + `residual_ids.json` (only valid residual rows)

These files are optional because peerfinder can still run from NPZ matrices.
FAISS GPU is used when available and enabled (`--faiss-gpu`).

## CLI reference

Core:

- `--filing_dir` default `sec_filings`
- `--out_dir` required
- `--filing` choices `10-K`, `10-Q`
- `--year` optional one or more years (default all years)
- `--scope` choices `heading`, `body`, `all`
- `--items` comma-separated item ids

Embedding:

- `--embedder` `local|openai`
- `--embed_model` (required for openai)
- `--batch_size`
- `--tokenizer-model`

Chunking:

- `--chunk_tokens` (default `280`)
- `--overlap_tokens` (default `60`)
- `--min_unit_tokens` (default `80`)
- `--min_units_per_item` (default `3`)

Decomposition:

- `--residual_norm_floor` (default `0.10`)
- `--cap_weight_percentile` (default `95.0`)

Indexing:

- `--build-faiss` / `--no-build-faiss`
- `--faiss-gpu` / `--no-faiss-gpu`

## Performance notes

- Most expensive steps:
  - embedding API calls (OpenAI mode),
  - serialization of large parquet with embedded vectors as JSON strings.
- local embedding is much faster but semantically weaker than foundation embeddings.
- FAISS build adds time but speeds large-scale nearest-neighbor usage.

## Typical commands

All-text scope with local embeddings:

```bash
python script/vdbbuilder.py --filing_dir sec_filings --out_dir vector_db --filing 10-K --year 2024 --scope all --embedder local
```

Heading-only scope:

```bash
python script/vdbbuilder.py --filing_dir sec_filings --out_dir vector_db --filing 10-K --year 2024 --scope heading --embedder local
```

Body-only scope with OpenAI:

```bash
set OPENAI_API_KEY=...
python script/vdbbuilder.py --filing_dir sec_filings --out_dir vector_db --filing 10-K --year 2024 --scope body --embedder openai --embed_model text-embedding-3-large
```

## Troubleshooting

1. `No source JSON filings found for scope=...`
- Check extractor outputs (`*_item.json` vs `*_str.json`) and `--scope`.
- Check `--filing_dir` root path.

2. `Filing directory does not exist` / `not a directory`
- Check `--filing_dir` value.

3. `No source JSON filings found for selected year(s): ...`
- Confirm requested years exist under `sec_filings/<cik>/<year>/...`.
- Remove `--year` to test full-directory discovery.

4. `No units/items built`
- Lower `--min_unit_tokens` or `--min_units_per_item`.
- Verify item ids passed in `--items` exist in extracted JSON.

5. OpenAI errors
- Verify `OPENAI_API_KEY`.
- Ensure `--embed_model` passed with `--embedder openai`.

6. FAISS GPU warning
- GPU may exist but faiss GPU bindings unavailable; install proper FAISS GPU package.
