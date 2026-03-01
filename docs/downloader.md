# Downloader Module Technical Guide

Module file: `script/downloader.py`

## Purpose

`downloader.py` retrieves SEC filings from EDGAR and stores:

- raw filing HTML (or source extension returned by EDGAR),
- metadata JSON per filing,
- run statistics and reports.

It supports either:

- all companies in selected years, or
- explicit company filters (`--ticker` or `--cik`).

## High-level flow

1. Parse CLI args and normalize filing code (`10k -> 10-K`).
2. Resolve ticker -> CIK mapping when needed.
3. Load SEC full-index records for target years.
4. Filter by:
   - selected CIKs (optional),
   - filing-date window per fiscal year (`--lookahead_months`).
5. For each matching record:
   - download filing by accession,
   - parse fiscal year tags from HTML,
   - write filing + meta under `<output_dir>/<cik>/<year>/<form>/`.
6. Write run-level reports under project `logs/` and `stats/`.

## Data contract

### Inputs

- SEC index records from `SECIndexParser`.
- EDGAR filing content from `SECDownloader.download_filing_by_accession`.

### Outputs

Per filing:

```text
<output_dir>/<cik>/<year>/<form>/<cik>_<year>_<form>.<ext>
<output_dir>/<cik>/<year>/<form>/<cik>_<year>_<form>_meta.json
```

Cross-run metadata:

```text
<output_dir>/_meta/cik_ticker_map_edgar.csv
<output_dir>/_meta/cik_ticker_map.csv
```

Run logs:

```text
logs/download_run_*.csv
stats/download_run_*.md
```

List-only mode:

```text
logs/list_only_*.csv
stats/list_only_*.md
```

## CLI reference

Required:

- `--filing`: one of `6k, 6ka, 8k, 8ka, 10q, 10qa, 10k, 10ka`
- `--year` or `--years`: one or more fiscal years
- `--output_dir`: output root
- `--user_agent`: SEC-compliant user-agent with contact

Optional:

- `--ticker <...>`: one or more tickers (mutually exclusive with `--cik`)
- `--cik <...>`: one or more CIKs
- `--lookahead_months` (alias `--lookahead_month`): default `12`
- `--overwrite`: overwrite existing local filing files
- `--list-only`: do not download, only produce counts/report

## Fiscal-year window logic

Filtering uses filing date, not report period.

Given fiscal year `FY` and `lookahead_months`:

- start = `FY-01-01`
- end = last day of month `12 + lookahead_months` relative to FY year

Special case in code:

- if `lookahead_months == 6`, end is `FY+1-06-30`.

## Metadata extraction from filing

Downloader attempts to parse from filing HTML:

- `dei:DocumentFiscalYearFocus`
- `dei:DocumentPeriodEndDate`
- `dei:TradingSymbol` (all unique symbols)

If fiscal year is missing after parsing, filing is counted as
`missing_fiscal_metadata` and skipped.

## Counters and statuses

Per run counters:

- `processed`
- `downloaded`
- `skipped_exists`
- `failed_download`
- `missing_fiscal_metadata`
- `skipped_outside_target_fy`

These are also broken down per requested fiscal year.

## Typical commands

Download all 10-K filings in fiscal year 2024 window:

```bash
python script/downloader.py --filing 10k --year 2024 --output_dir sec_filings --user_agent "RAG4PeerFirm/1.0 (you@example.com)"
```

Ticker-filtered:

```bash
python script/downloader.py --filing 10k --year 2024 --ticker AAPL MSFT --output_dir sec_filings --user_agent "RAG4PeerFirm/1.0 (you@example.com)"
```

List-only (no network download):

```bash
python script/downloader.py --filing 10k --year 2023 2024 --list-only --output_dir sec_filings --user_agent "RAG4PeerFirm/1.0 (you@example.com)"
```

## Failure modes and troubleshooting

1. `Unsupported --filing ...`
- Use short code format from supported list.

2. SEC blocking / bad user-agent
- Ensure `--user_agent` includes app + email contact.

3. Missing filings for expected firms
- Increase `--lookahead_months`.
- Check ticker to CIK mapping and CIK zero-padding.

4. Many `missing_fiscal_metadata`
- Filing HTML may lack required DEI tags or parser may not match variants.

5. Existing files not replaced
- Add `--overwrite`.

