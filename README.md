# RAG4PeerFirm

Pipeline for SEC filing download, item extraction, vector building, and peer-firm matching.

## Modules

- `script/downloader.py`
  - Downloads SEC filings using ticker/CIK filters into `sec_filings/`.
- `script/vdbbuilder.py`
  - Builds unit and item embeddings from extracted filing JSON (`*_item.json` or `*_str.json` by scope).
  - Produces pooled/residual vectors and optional FAISS indices.
- `script/peerfinder.py`
  - Finds item-level peers for a focal firm and year.
  - Supports `orthogonal` and `cosine` similarity methods.
  - Supports cached precomputed NxN similarity matrices.

## Input Data Assumptions

- Input format is JSON-only from extracted filing outputs.
- Expected location pattern:
  - `sec_filings/<firm_id>/<year>/<filing_type>/<firm_id>_<year>_<filing>_item.json`
- Expected JSON keys:
  - `toc_items`
  - `items[item_id].text_content` (fallback to `html_content`)

## Install

```bash
pip install -r requirements.txt
```

## Testing

Run integration tests:

```bash
python -m pytest tests/test_integration_pipeline.py
```

Run only vdbbuilder coverage:

```bash
python -m pytest tests/test_integration_pipeline.py -k vdbbuilder
```

Optional local marker registration (to suppress `PytestUnknownMarkWarning` for `integration`):

```ini
# pytest.ini (local file)
[pytest]
markers =
    integration: integration tests for end-to-end pipeline behavior
```

## 1) Download Filings

```bash
python script/downloader.py \
  --filing 10k \
  --year 2024 \
  --output_dir sec_filings \
  --user_agent "RAG4PeerFirm/1.0 (your-email@example.com)"
```

Examples:

```bash
python script/downloader.py --filing 10k --year 2024 --ticker AAPL MSFT --output_dir sec_filings --user_agent "RAG4PeerFirm/1.0 (your-email@example.com)"
python script/downloader.py --filing 10k --year 2024 --cik 0000320193 0000789019 --output_dir sec_filings --user_agent "RAG4PeerFirm/1.0 (your-email@example.com)"
```

## 2) Build Vector DB

Default local embedder (deterministic and offline):

```bash
python script/vdbbuilder.py \
  --filing_dir sec_filings \
  --out_dir vector_db \
  --filing 10-K \
  --year 2024 \
  --scope all \
  --embedder local
```

OpenAI embedder:

```bash
set OPENAI_API_KEY=...
python script/vdbbuilder.py \
  --filing_dir sec_filings \
  --out_dir vector_db \
  --filing 10-K \
  --year 2024 \
  --scope all \
  --embedder openai \
  --embed_model text-embedding-3-large
```

Scope options:

- `--scope all`: use `*_item.json` and `items[item_id].text_content` (existing behavior)
- `--scope heading`: use `*_str.json` and concatenate all `heading` values in each item structure
- `--scope body`: use `*_str.json` and concatenate all `body` values in each item structure
- `--year` (optional): one or more years to build
  - default: build all years found under `--filing_dir`
  - example: `--year 2012 2013`

Scope-specific outputs are written under:

- `vector_db/scope=<scope>/...`

Optional flags:

```bash
python script/vdbbuilder.py --out_dir vector_db --embedder local --no-build-faiss
python script/vdbbuilder.py --out_dir vector_db --embedder local --no-faiss-gpu
```

Main outputs:

- `vector_db/scope=<scope>/units/units_<YEAR>.parquet`
- `vector_db/scope=<scope>/item_vectors/item_vectors_<YEAR>.parquet`
- `vector_db/scope=<scope>/vectors/pooled/item=<ITEM>/year=<YEAR>.npz`
- `vector_db/scope=<scope>/vectors/residual/item=<ITEM>/year=<YEAR>.npz`
- `vector_db/scope=<scope>/indices/item=<ITEM>/year=<YEAR>/...` (if FAISS build is enabled)

Error behavior:

- errors if `--filing_dir` does not exist
- errors if `--filing_dir` exists but has no matching source JSON for selected scope/filing
- errors if selected `--year` values have no matching filings

## 3) Find Peers

Orthogonal method:

```bash
python script/peerfinder.py \
  --vdb_dir vector_db \
  --scope all \
  --focalfirm 0000320193 \
  --year 2024 \
  --item 1A \
  --method orthogonal \
  --k 20 \
  --q_share 0.20 \
  --out_path output/peer_sets.csv.gz
```

Pairwise cosine method:

```bash
python script/peerfinder.py \
  --vdb_dir vector_db \
  --scope all \
  --focalfirm 0000320193 \
  --year 2024 \
  --item 1A \
  --method cosine \
  --k 20
```

Multiple items:

```bash
python script/peerfinder.py --vdb_dir vector_db --scope all --focalfirm 0000320193 --year 2024 --item 1A 7 7A --method orthogonal
```

Gemini text-comparison method (uses item text from `units.parquet`):

```bash
set GEMINI_API_KEY=...
python script/peerfinder.py \
  --vdb_dir vector_db \
  --scope all \
  --focalfirm 0000320193 \
  --year 2024 \
  --item 1A \
  --method gemini \
  --gemini-model gemini-3-flash-preview \
  --k 20
```

Gemini free-tier pacing defaults are built in:

- `--gemini-rpm 5`
- `--gemini-tpm 250000`
- `--gemini-rpd 20`

Reliability controls for Gemini:

- `--gemini-max-retries` (default `5`)
- `--gemini-backoff-base-sec` (default `2.0`)
- `--gemini-timeout-sec` (default `90`)

## Precompute Similarity Matrix Cache

Build and save NxN similarity matrix per `(item, year, method)`:

```bash
python script/peerfinder.py \
  --vdb_dir vector_db \
  --focalfirm 0000320193 \
  --year 2024 \
  --item 1A \
  --method orthogonal \
  --precompute
```

Force rebuild:

```bash
python script/peerfinder.py --vdb_dir vector_db --focalfirm 0000320193 --year 2024 --item 1A --method orthogonal --precompute --precompute-overwrite
```

Cache behavior:

- If table exists, `peerfinder` uses it first.
- If table does not exist, `peerfinder` computes on demand.
- If `--precompute` is passed, missing table is created and then used.

Cache paths:

- `vector_db/scope=<scope>/precomputed/scope=<scope>/item=<ITEM>/year=<YEAR>/method=<METHOD>/similarity.npy`
- `vector_db/scope=<scope>/precomputed/scope=<scope>/item=<ITEM>/year=<YEAR>/method=<METHOD>/firm_ids.json`
