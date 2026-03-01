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

### `units.parquet`

One row per unit:

- `firm_id`, `year`, `item_id`, `unit_id`
- `unit_text`, `unit_tokens`
- `embedding_model`
- `embedding` (JSON-serialized vector)
- `source_path`, `scope`

### `item_vectors.parquet`

One row per `(firm_id, year, item_id)`:

- `num_units`, `item_tokens`
- `pooled_embedding` (JSON)
- `w_max`, `w_mean`
- `common_loading`, `residual_norm`, `residual_embedding`
- `scope`

### NPZ matrices

- `vectors/pooled/item=<ITEM>/year=<YEAR>.npz`
  - `mat`: pooled matrix `[N, D]`
  - `ids`: aligned firm ids `[N]`

- `vectors/residual/item=<ITEM>/year=<YEAR>.npz`
  - `mat`: residual matrix `[N, D]` (zero rows for invalid residuals)
  - `ids`: aligned firm ids `[N]`
  - `mask`: residual valid flag `[N]` (`1` valid, `0` invalid)

### Optional FAISS indices

Written if `--build-faiss`:

- pooled index + ids
- residual index + ids (only valid residual rows)

FAISS GPU is used when available and enabled (`--faiss-gpu`).

## CLI reference

Core:

- `--filing_dir` default `sec_filings`
- `--out_dir` required
- `--filing` choices `10-K`, `10-Q`
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
python script/vdbbuilder.py --filing_dir sec_filings --out_dir vector_db --filing 10-K --scope all --embedder local
```

Heading-only scope:

```bash
python script/vdbbuilder.py --filing_dir sec_filings --out_dir vector_db --filing 10-K --scope heading --embedder local
```

Body-only scope with OpenAI:

```bash
set OPENAI_API_KEY=...
python script/vdbbuilder.py --filing_dir sec_filings --out_dir vector_db --filing 10-K --scope body --embedder openai --embed_model text-embedding-3-large
```

## Troubleshooting

1. `No source JSON filings found for scope=...`
- Check extractor outputs (`*_item.json` vs `*_str.json`) and `--scope`.
- Check `--filing_dir` root path.

2. `No units/items built`
- Lower `--min_unit_tokens` or `--min_units_per_item`.
- Verify item ids passed in `--items` exist in extracted JSON.

3. OpenAI errors
- Verify `OPENAI_API_KEY`.
- Ensure `--embed_model` passed with `--embedder openai`.

4. FAISS GPU warning
- GPU may exist but faiss GPU bindings unavailable; install proper FAISS GPU package.
