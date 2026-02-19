# Full-Index Download Feature - Implementation Summary

## Overview
Added capability to download ALL companies' filings from SEC EDGAR when no `--tickers` or `--ciks` are specified. Uses SEC's quarterly full-index files to discover all companies.

## Feature Details

### How It Works
1. **Index Source**: Downloads `company.idx` files from `https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{1-4}/`
2. **Parsing**: Extracts company information from fixed-width or pipe-separated format
3. **Filtering**: Filters by filing type (10-K, 10-Q) 
4. **Deduplication**: Removes duplicate (CIK, year) combinations
5. **Extraction**: Downloads and extracts items from all discovered filings

### Usage Examples

#### Download ALL 10-K filings for 2024
```bash
python main.py --filing 10-K --year 2024
```

#### Download ALL 10-K filings for multiple years
```bash
python main.py --filing 10-K --years 2023 2024 2025
```

#### Download ALL companies but specific items only
```bash
python main.py --filing 10-K --year 2024 --items 1 1A 7
```

#### Extract structure from all companies
```bash
# Step 1: Download all filings
python main.py --filing 10-K --year 2024

# Step 2: Extract structure
python main.py --filing 10-K --year 2024 --extract-structure
```

## Scale & Performance

### Expected Volume (2024 data)
- **10-K filings**: ~6,700 per year
- **10-Q filings**: ~20,000 per year (all 3 quarters)
- **Multiple years**: Linear scaling (e.g., 3 years = ~20,000 10-Ks)

### Time Estimates
- **Index Download**: ~0.4 seconds per quarter (1.6s per year)
- **Filing Download**: ~0.2 seconds per filing (with rate limiting)
- **Total for 6,700 10-Ks**: ~22 minutes minimum (download only)
- **With extraction**: 2-3 hours for full processing

### Storage Requirements
- **Per 10-K filing**: ~500KB - 2MB
- **6,700 filings**: ~3-13 GB
- **Extracted items**: Additional 50-200 MB per 1,000 filings

## Safety Features

### 1. Confirmation Prompt
When no companies specified, displays:
```
================================================================================
WARNING: No tickers or CIKs specified - will download ALL companies!
================================================================================

Estimated 10-K filings: ~4,500
Years: 2024
Quarters to check: 4

This may take several hours and require significant storage.
Rate limiting: 10 requests/second (SEC requirement)

Do you want to continue? (yes/no):
```

### 2. Progress Reporting
- Shows companies found per filing type
- Displays total unique companies
- Reports quarters scanned

### 3. Rate Limiting
- Respects SEC's 10 requests/second limit
- Built-in delays between requests
- Prevents IP blocking

## Implementation Files

### New File: `src/index_parser.py` (212 lines)
**Classes:**
- `SECIndexParser`: Main class for parsing full-index files

**Key Methods:**
- `_download_index_file(year, quarter)`: Downloads company.idx
- `_parse_index_file(content, filing_type)`: Parses index format
- `get_all_companies_for_filing(filing_type, years)`: Returns all filings
- `get_ciks_for_filing(filing_type, years)`: Returns unique CIKs
- `estimate_filing_count(filing_type, years)`: Estimates volume

### Modified: `main.py`
**Changes:**
1. Import `SECIndexParser`
2. Initialize `self.index_parser` in `__init__`
3. Make `--ticker`/`--cik` optional (not required)
4. Add detection logic for missing companies
5. Add confirmation prompt
6. Fetch CIKs from index when needed
7. Update help examples

## Testing Results

### Test 1: Index Parsing (2024 10-K)
```
Found 6,768 unique 10-K filings in 2024
Time elapsed: 1.65 seconds
```

### Test 2: Specific Company Verification
```
✓ Apple (CIK 0000320193) found in 2024 10-K filings
  Company: Apple Inc.
  Date Filed: 2024-11
```

### Test 3: Multi-year Estimation
```
Estimated 10-K filings for 2023-2024: 9,000
Quarters to check: 8
```

## Index File Format

### Fixed-Width Format (Older Years)
```
Column Positions:
  Company Name: 0-62
  Form Type:    62-74
  CIK:          74-86
  Date Filed:   86-98
  File Name:    98+
```

### Pipe-Separated Format (Recent Years)
```
Company Name|Form Type|CIK|Date Filed|File Name
Apple Inc.|10-K|0000320193|2024-11-01|edgar/data/320193/0000320193-24-000123.txt
```

## Error Handling

### Graceful Failures
- Future quarters return empty (404) → Skip gracefully
- Network errors → Log warning, continue with other quarters
- Parse errors → Skip malformed lines
- Missing companies → Continue processing valid entries

### Deduplication
- Tracks `(CIK, year)` combinations
- Prevents duplicate downloads
- Maintains most recent filing per company/year

## Best Practices

### For Large Downloads
1. **Use screen/tmux** for long-running processes
2. **Monitor disk space** before starting
3. **Check network stability** for multi-hour downloads
4. **Start with single year** to test before scaling
5. **Use --items** to limit extraction scope

### For Testing
```bash
# Test with single year first
python main.py --filing 10-K --year 2024 --items 1

# Then scale to multiple years
python main.py --filing 10-K --years 2023 2024 2025
```

### For Production
```bash
# Use high worker count for parallel processing
python main.py --filing 10-K --year 2024 --workers 16

# Separate structure extraction after download
python main.py --filing 10-K --year 2024 --extract-structure --workers 16
```

## Limitations

1. **No filtering by industry/sector**: Downloads all companies
2. **No date range within year**: Full year only
3. **Amended filings**: May include duplicates (10-K/A)
4. **Foreign filers**: Includes all filers (F-10, 20-F not filtered)

## Future Enhancements

Potential improvements:
- Add `--min-date` / `--max-date` filters
- Add `--industry` / `--sector` filters
- Add `--exclude-amendments` flag
- Add progress bar for large downloads
- Add resume capability for interrupted downloads
- Export CIK list to file before downloading
