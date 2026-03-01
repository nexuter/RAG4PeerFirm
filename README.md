# RAG4PeerFirm

Minimal project for SEC filing download, vector building, and peer identification.

## Project Layout

- `script/downloader.py`: SEC EDGAR downloader client (supports ticker/CIK and accession-based download)
- `utils/index_parser.py`: SEC full-index parser for filing records and CIK lists
- `utils/file_manager.py`: file IO helpers for filing storage
- `script/vdbbuilder.py`: builds unit table, item vectors, residual vectors, and per-item/year FAISS indices
- `script/peerfinder.py`: common-screen + specific-rank peer search using built vectors/indices
- `test/smoke_test.py`: minimal import/API smoke test for core modules

## Install

```bash
pip install -r requirements.txt
```

## Smoke Test

```bash
python test/smoke_test.py
```

## 1) Download Filings

Use your downloader entrypoint/CLI to populate raw filing data under your filing directory.

Expected directory shape for downstream steps:

```text
<filing_dir>/<cik>/<year>/<filing_type>/<file>.txt|.htm|.html
```

## 2) Build Vector DB

```bash
python script/vdbbuilder.py \
  --filing_dir <filing_dir> \
  --out_dir <vdb_out> \
  --filing 10-K \
  --embedder local
```

OpenAI embeddings:

```bash
set OPENAI_API_KEY=...    # Windows
python script/vdbbuilder.py --filing_dir <filing_dir> --out_dir <vdb_out> --embedder openai
```

Outputs under `<vdb_out>`:

- `units.parquet`
- `item_vectors.parquet`
- `vectors/pooled/item=*/year=*.npz`
- `vectors/residual/item=*/year=*.npz`
- `indices/item=*/year=*/pooled.faiss` (+ residual where available)

## 3) Find Peers

```bash
python script/peerfinder.py \
  --vdb_dir <vdb_out> \
  --focalfirm <cik_or_firm_id> \
  --year <year> \
  --item 1A \
  --k 20 \
  --q_share 0.20 \
  --out_path output/peer_sets.csv.gz
```

Use `--item all` to run across all available items.

## Notes

- `peerfinder` requires FAISS.
- `vdbbuilder --embedder local` is deterministic and offline-friendly.


