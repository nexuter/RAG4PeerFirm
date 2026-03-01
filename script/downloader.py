"""
Primary filing downloader (EDGAR only).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.downloader import SECDownloader
from utils.index_parser import SECIndexParser
from utils.file_manager import FileManager


FILING_CODE_MAP: Dict[str, Tuple[str, str]] = {
    "6k": ("6-K", "6-K"),
    "6ka": ("6-K/A", "6-KA"),
    "8k": ("8-K", "8-K"),
    "8ka": ("8-K/A", "8-KA"),
    "10q": ("10-Q", "10-Q"),
    "10qa": ("10-Q/A", "10-QA"),
    "10k": ("10-K", "10-K"),
    "10ka": ("10-K/A", "10-KA"),
}


def _get_filtered_records(
    *,
    sec_form: str,
    fiscal_years: List[int],
    lookahead_months: int,
    target_ciks: Set[str],
) -> List[Dict[str, str]]:
    index_parser = SECIndexParser()
    index_years = sorted(set(y for fy in fiscal_years for y in (fy, fy + 1)))
    records = index_parser.get_filing_records_for_filing(sec_form, index_years)
    print(f"Loaded index records: {len(records)} ({sec_form}, years={index_years})")

    unique_by_accession: Dict[str, Dict[str, str]] = {}
    for r in records:
        acc = r.get("accession_number", "")
        if acc:
            unique_by_accession[acc] = r
    deduped_records = list(unique_by_accession.values())
    print(f"Unique records by accession: {len(deduped_records)}")

    filtered_records: List[Dict[str, str]] = []
    for r in deduped_records:
        cik = (r.get("cik_padded") or "").zfill(10)
        if target_ciks and cik not in target_ciks:
            continue
        fd = r.get("date_filed", "")
        if any(_in_window_for_fiscal_year(fd, fy, lookahead_months) for fy in fiscal_years):
            filtered_records.append(r)
    print(f"Records in scope after filters: {len(filtered_records)}")
    return filtered_records


def _write_list_only_report(
    *,
    sec_form: str,
    fiscal_years: List[int],
    lookahead_months: int,
    filtered_records: List[Dict[str, str]],
) -> None:
    logs_dir = PROJECT_ROOT / "logs"
    stats_dir = PROJECT_ROOT / "stats"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows: List[Dict[str, object]] = []
    for fy in fiscal_years:
        ciks = set()
        filings = 0
        for r in filtered_records:
            fd = r.get("date_filed", "")
            if _in_window_for_fiscal_year(fd, fy, lookahead_months):
                filings += 1
                ciks.add((r.get("cik_padded") or "").zfill(10))
        rows.append({"fiscal_year": fy, "cik_count": len(ciks), "filing_count": filings})

    out_csv = logs_dir / f"list_only_{sec_form.lower().replace('/', '').replace('-', '')}_{stamp}.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fiscal_year", "cik_count", "filing_count"])
        w.writeheader()
        for row in rows:
            w.writerow(row)

    out_md = stats_dir / f"list_only_{sec_form.lower().replace('/', '').replace('-', '')}_{stamp}.md"
    lines = [
        "# EDGAR List-Only Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"- Filing form: `{sec_form}`",
        f"- Fiscal years: `{fiscal_years}`",
        f"- Lookahead months: `{lookahead_months}`",
        "",
        "## Summary",
        "| Fiscal Year | CIK Count | Filing Count |",
        "|---|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['fiscal_year']} | {row['cik_count']} | {row['filing_count']} |")
    lines.extend(
        [
            "",
            "## Artifact",
            f"- CSV: `{out_csv}`",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"List-only report saved: {out_md}")
    print(f"List-only csv saved: {out_csv}")


def _write_download_run_report(
    *,
    sec_form: str,
    fiscal_years: List[int],
    lookahead_months: int,
    output_dir: Path,
    stats_total: Dict[str, int],
    stats_by_year: Dict[int, Dict[str, int]],
) -> None:
    logs_dir = PROJECT_ROOT / "logs"
    stats_dir = PROJECT_ROOT / "stats"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    code = sec_form.lower().replace("/", "").replace("-", "")

    csv_path = logs_dir / f"download_run_{code}_{stamp}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "fiscal_year",
                "downloaded",
                "skipped_exists",
                "missing_fiscal_metadata",
                "failed_download",
                "skipped_outside_target_fy",
            ],
        )
        w.writeheader()
        for fy in fiscal_years:
            row = stats_by_year.get(fy, {})
            w.writerow(
                {
                    "fiscal_year": fy,
                    "downloaded": row.get("downloaded", 0),
                    "skipped_exists": row.get("skipped_exists", 0),
                    "missing_fiscal_metadata": row.get("missing_fiscal_metadata", 0),
                    "failed_download": row.get("failed_download", 0),
                    "skipped_outside_target_fy": row.get("skipped_outside_target_fy", 0),
                }
            )

    md_path = stats_dir / f"download_run_{code}_{stamp}.md"
    lines = [
        "# Download Run Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"- Filing form: `{sec_form}`",
        f"- Fiscal years: `{fiscal_years}`",
        f"- Lookahead months: `{lookahead_months}`",
        f"- Output dir: `{output_dir}`",
        "",
        "## Totals",
        f"- Processed: `{stats_total.get('processed', 0)}`",
        f"- Downloaded: `{stats_total.get('downloaded', 0)}`",
        f"- Skipped exists: `{stats_total.get('skipped_exists', 0)}`",
        f"- Failed download: `{stats_total.get('failed_download', 0)}`",
        f"- Missing fiscal metadata: `{stats_total.get('missing_fiscal_metadata', 0)}`",
        f"- Skipped outside target FY: `{stats_total.get('skipped_outside_target_fy', 0)}`",
        "",
        "## Yearly Measurements",
        "| Fiscal Year | Downloaded | Skipped Exists | Missing Fiscal Metadata | Failed Download | Skipped Outside Target FY |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for fy in fiscal_years:
        row = stats_by_year.get(fy, {})
        lines.append(
            f"| {fy} | {row.get('downloaded', 0)} | {row.get('skipped_exists', 0)} | "
            f"{row.get('missing_fiscal_metadata', 0)} | {row.get('failed_download', 0)} | "
            f"{row.get('skipped_outside_target_fy', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Artifact",
            f"- CSV: `{csv_path}`",
            "",
            f"search filings looking ahead {lookahead_months} months",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Download run report saved: {md_path}")
    print(f"Download run csv saved: {csv_path}")


def _report_progress(
    *,
    processed: int,
    total: int,
    start_time: float,
    stats: Dict[str, int],
) -> None:
    if processed <= 0 or total <= 0:
        return
    elapsed = time.monotonic() - start_time
    avg = elapsed / processed
    expected_total = avg * total
    eta = max(expected_total - elapsed, 0.0)
    pct = (processed / total) * 100.0
    print(
        f"[EDGAR] {processed}/{total} ({pct:.1f}%) | "
        f"elapsed {elapsed/60.0:.1f}m / expected {expected_total/60.0:.1f}m | "
        f"ETA {eta/60.0:.1f}m | "
        f"downloaded={stats.get('downloaded', 0)} "
        f"exists={stats.get('skipped_exists', 0)} "
        f"failed={stats.get('failed_download', 0)}"
    )


def _extract_dual_dates(html_content: str) -> Tuple[Optional[str], Optional[int], Dict[str, str]]:
    tags_found: Dict[str, str] = {}

    fy_match = re.search(
        r'name="dei:DocumentFiscalYearFocus"[^>]*>\s*([12]\d{3})\s*<',
        html_content,
        flags=re.IGNORECASE,
    )
    fiscal_year = int(fy_match.group(1)) if fy_match else None
    if fy_match:
        tags_found["dei:DocumentFiscalYearFocus"] = fy_match.group(1)

    period_match = re.search(
        r'name="dei:DocumentPeriodEndDate"[^>]*>\s*([12]\d{3}-\d{2}-\d{2})\s*<',
        html_content,
        flags=re.IGNORECASE,
    )
    period_of_report = period_match.group(1) if period_match else None
    if period_match:
        tags_found["dei:DocumentPeriodEndDate"] = period_match.group(1)

    if fiscal_year is None and period_of_report:
        fiscal_year = int(period_of_report[:4])

    return period_of_report, fiscal_year, tags_found


def _extract_trading_symbols(html_content: str) -> List[str]:
    symbols = re.findall(
        r'name="dei:TradingSymbol"[^>]*>\s*([A-Za-z0-9\.\-]+)\s*<',
        html_content,
        flags=re.IGNORECASE,
    )
    out: List[str] = []
    seen: Set[str] = set()
    for sym in symbols:
        token = sym.strip().upper()
        if token and token not in seen:
            seen.add(token)
            out.append(token)
    return out


def _parse_filing_date(value: str) -> Optional[date]:
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y%m%d"):
        try:
            parsed = datetime.strptime(value, fmt).date()
            if fmt == "%Y-%m":
                parsed = parsed.replace(day=1)
            return parsed
        except ValueError:
            continue
    return None


def _in_window_for_fiscal_year(filing_date: str, fiscal_year: int, lookahead_months: int) -> bool:
    filed = _parse_filing_date(filing_date)
    if not filed:
        return False

    start = date(fiscal_year, 1, 1)
    if lookahead_months == 6:
        end = date(fiscal_year + 1, 6, 30)
    else:
        month = 12 + lookahead_months
        end_year = fiscal_year + (month - 1) // 12
        end_month = ((month - 1) % 12) + 1
        if end_month in {1, 3, 5, 7, 8, 10, 12}:
            end_day = 31
        elif end_month in {4, 6, 9, 11}:
            end_day = 30
        else:
            leap = (end_year % 4 == 0 and end_year % 100 != 0) or (end_year % 400 == 0)
            end_day = 29 if leap else 28
        end = date(end_year, end_month, end_day)
    return start <= filed <= end


def _normalize_cik_set(
    downloader: SECDownloader,
    tickers: Optional[List[str]],
    ciks: Optional[List[str]],
) -> Set[str]:
    result: Set[str] = set()
    for c in ciks or []:
        token = c.strip()
        if token:
            result.add(token.zfill(10))
    for t in tickers or []:
        token = t.strip()
        if not token:
            continue
        if token.isdigit():
            result.add(token.zfill(10))
        else:
            result.add(downloader.get_cik(token))
    return result


def _normalize_ticker_input_map(
    downloader: SECDownloader,
    tickers: Optional[List[str]],
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for token in tickers or []:
        t = token.strip()
        if not t or t.isdigit():
            continue
        cik = downloader.get_cik(t)
        out[cik] = t.upper()
    return out


def _load_cik_ticker_map(path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    rows: Dict[Tuple[str, str], Dict[str, str]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            year = str(r.get("fiscal_year", "")).strip()
            cik = str(r.get("cik", "")).strip().zfill(10)
            if not year or not cik:
                continue
            rows[(year, cik)] = {
                "fiscal_year": year,
                "cik": cik,
                "ticker": str(r.get("ticker", "")).strip().upper(),
                "source": "edgar",
                "updated_at": str(r.get("updated_at", "")).strip(),
            }
    return rows


def _save_cik_ticker_map(path: Path, rows: Dict[Tuple[str, str], Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows.values(), key=lambda r: (r["fiscal_year"], r["cik"]))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["fiscal_year", "cik", "ticker", "source", "updated_at"],
        )
        writer.writeheader()
        writer.writerows(ordered)


def _upsert_cik_ticker_map(
    path: Path,
    *,
    fiscal_year: int,
    cik: str,
    ticker: Optional[str],
) -> None:
    if not ticker:
        return
    t = ticker.strip().upper()
    if not t:
        return
    rows = _load_cik_ticker_map(path)
    key = (str(fiscal_year), cik.zfill(10))
    rows[key] = {
        "fiscal_year": str(fiscal_year),
        "cik": cik.zfill(10),
        "ticker": t,
        "source": "edgar",
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_cik_ticker_map(path, rows)


def _save_filing_and_meta(
    fm: FileManager,
    output_dir: Path,
    cik: str,
    fiscal_year: int,
    folder_form: str,
    extension: str,
    html_content: str,
    meta: Dict[str, object],
    overwrite: bool,
) -> str:
    filing_dir = output_dir / cik / str(fiscal_year) / folder_form
    filing_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{cik}_{fiscal_year}_{folder_form}"
    filing_path = filing_dir / f"{base_name}.{extension}"
    meta_path = filing_dir / f"{base_name}_meta.json"

    if filing_path.exists() and not overwrite:
        return "skipped_exists"

    fm.save_html(str(filing_path), html_content)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return "downloaded"


def download_from_edgar(
    *,
    sec_form: str,
    folder_form: str,
    fiscal_years: List[int],
    output_dir: Path,
    lookahead_months: int,
    tickers: Optional[List[str]],
    ciks: Optional[List[str]],
    overwrite: bool,
    user_agent: str,
) -> None:
    downloader = SECDownloader(user_agent=user_agent)
    fm = FileManager(str(output_dir))
    map_path = output_dir / "_meta" / "cik_ticker_map_edgar.csv"
    legacy_map_path = output_dir / "_meta" / "cik_ticker_map.csv"
    input_ticker_by_cik = _normalize_ticker_input_map(downloader, tickers)

    if date.today().month <= lookahead_months:
        print(
            f"DISCLAIMER: Current date is within first {lookahead_months} months. "
            "Some companies may not have filed latest disclosures yet."
        )

    target_ciks = _normalize_cik_set(downloader, tickers, ciks)
    if target_ciks:
        print(f"CIK filter enabled: {len(target_ciks)} target CIK(s)")
    else:
        print("CIK filter disabled: processing all CIKs in selected years/window")

    filtered_records = _get_filtered_records(
        sec_form=sec_form,
        fiscal_years=fiscal_years,
        lookahead_months=lookahead_months,
        target_ciks=target_ciks,
    )

    stats = {
        "processed": 0,
        "downloaded": 0,
        "skipped_exists": 0,
        "failed_download": 0,
        "missing_fiscal_metadata": 0,
        "skipped_outside_target_fy": 0,
    }
    stats_by_year: Dict[int, Dict[str, int]] = {
        fy: {
            "downloaded": 0,
            "skipped_exists": 0,
            "missing_fiscal_metadata": 0,
            "failed_download": 0,
            "skipped_outside_target_fy": 0,
        }
        for fy in fiscal_years
    }
    total = len(filtered_records)
    start_time = time.monotonic()
    update_every = 1 if total <= 200 else max(1, total // 200)

    for i, record in enumerate(filtered_records, start=1):
        stats["processed"] += 1
        cik = (record.get("cik_padded") or "").zfill(10)
        accession = record.get("accession_number", "")
        filing_date = record.get("date_filed", "")
        status_prefix = f"[{i}/{total}] cik={cik} accession={accession} filed={filing_date}"
        if not accession:
            stats["failed_download"] += 1
            print(f"{status_prefix} result=failed_download reason=missing_accession")
            continue

        try:
            html_content, ext, normalized_cik = downloader.download_filing_by_accession(cik, accession)
        except Exception:
            stats["failed_download"] += 1
            # Filing-date window already matched at least one target FY.
            for fy in fiscal_years:
                if _in_window_for_fiscal_year(filing_date, fy, lookahead_months):
                    stats_by_year[fy]["failed_download"] += 1
            print(f"{status_prefix} result=failed_download")
            continue

        period_of_report, fiscal_year, tags_found = _extract_dual_dates(html_content)
        symbols = _extract_trading_symbols(html_content)
        if fiscal_year is None:
            stats["missing_fiscal_metadata"] += 1
            for fy in fiscal_years:
                if _in_window_for_fiscal_year(filing_date, fy, lookahead_months):
                    stats_by_year[fy]["missing_fiscal_metadata"] += 1
            print(f"{status_prefix} result=missing_fiscal_metadata")
            continue
        if fiscal_year not in fiscal_years:
            stats["skipped_outside_target_fy"] += 1
            print(f"{status_prefix} result=skipped_outside_target_fy fiscal_year={fiscal_year}")
            continue

        meta = {
            "source": "edgar",
            "cik": normalized_cik,
            "fiscal_year": fiscal_year,
            "filing_type": sec_form,
            "folder_form": folder_form,
            "filing_date": filing_date,
            "period_of_report": period_of_report,
            "accession_number": accession,
            "source_file_name": record.get("file_name", ""),
            "tags_found": tags_found,
            "ticker_symbols": symbols,
        }
        state = _save_filing_and_meta(
            fm=fm,
            output_dir=output_dir,
            cik=normalized_cik,
            fiscal_year=fiscal_year,
            folder_form=folder_form,
            extension=ext,
            html_content=html_content,
            meta=meta,
            overwrite=overwrite,
        )
        if state == "downloaded":
            stats["downloaded"] += 1
            stats_by_year[fiscal_year]["downloaded"] += 1
            print(f"{status_prefix} result=downloaded fiscal_year={fiscal_year}")
        else:
            stats["skipped_exists"] += 1
            stats_by_year[fiscal_year]["skipped_exists"] += 1
            print(f"{status_prefix} result=skipped_exists fiscal_year={fiscal_year}")

        preferred_ticker = symbols[0] if symbols else input_ticker_by_cik.get(normalized_cik)
        _upsert_cik_ticker_map(
            map_path,
            fiscal_year=fiscal_year,
            cik=normalized_cik,
            ticker=preferred_ticker,
        )
        _upsert_cik_ticker_map(
            legacy_map_path,
            fiscal_year=fiscal_year,
            cik=normalized_cik,
            ticker=preferred_ticker,
        )

        if (i % update_every == 0) or (i == total):
            _report_progress(
                processed=i,
                total=total,
                start_time=start_time,
                stats=stats,
            )

    print("EDGAR download completed.")
    print(json.dumps(stats, indent=2))
    _write_download_run_report(
        sec_form=sec_form,
        fiscal_years=fiscal_years,
        lookahead_months=lookahead_months,
        output_dir=output_dir,
        stats_total=stats,
        stats_by_year=stats_by_year,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download filings from EDGAR.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--ticker", nargs="+", dest="tickers", help="Ticker(s) to download.")
    group.add_argument("--cik", nargs="+", dest="ciks", help="CIK(s) to download.")
    parser.add_argument(
        "--filing",
        required=True,
        help="Filing code: 6k, 6ka, 8k, 8ka, 10q, 10qa, 10k, 10ka",
    )
    parser.add_argument("--year", "--years", nargs="+", required=True, dest="years", help="Fiscal year(s).")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument(
        "--lookahead_month",
        "--lookahead_months",
        type=int,
        default=12,
        dest="lookahead_months",
        help="Lookahead months for dual-date filtering (default: 12).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing filings.")
    parser.add_argument("--list-only", action="store_true", help="Build fiscal-year filing list only (no download).")
    parser.add_argument(
        "--user_agent",
        required=True,
        help="Custom SEC User-Agent string (include contact email).",
    )
    args = parser.parse_args()

    filing_key = args.filing.strip().lower()
    if filing_key not in FILING_CODE_MAP:
        parser.error(f"Unsupported --filing `{args.filing}`.")
    sec_form, folder_form = FILING_CODE_MAP[filing_key]

    fiscal_years = sorted({int(y) for y in args.years})
    if any(y < 1995 or y > 2100 for y in fiscal_years):
        parser.error("Fiscal year must be between 1995 and 2100.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloader = SECDownloader(user_agent=args.user_agent)
    target_ciks = _normalize_cik_set(downloader, args.tickers, args.ciks)

    if args.list_only:
        if target_ciks:
            print(f"CIK filter enabled: {len(target_ciks)} target CIK(s)")
        else:
            print("CIK filter disabled: processing all CIKs in selected years/window")
        filtered_records = _get_filtered_records(
            sec_form=sec_form,
            fiscal_years=fiscal_years,
            lookahead_months=args.lookahead_months,
            target_ciks=target_ciks,
        )
        _write_list_only_report(
            sec_form=sec_form,
            fiscal_years=fiscal_years,
            lookahead_months=args.lookahead_months,
            filtered_records=filtered_records,
        )
        return

    download_from_edgar(
        sec_form=sec_form,
        folder_form=folder_form,
        fiscal_years=fiscal_years,
        output_dir=output_dir,
        lookahead_months=args.lookahead_months,
        tickers=args.tickers,
        ciks=args.ciks,
        overwrite=args.overwrite,
        user_agent=args.user_agent,
    )


if __name__ == "__main__":
    main()
