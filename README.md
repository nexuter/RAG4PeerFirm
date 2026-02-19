# ItemXtractor

A professional Python tool for extracting specific items from SEC EDGAR 10-K and 10-Q filings. ItemXtractor automatically downloads filings, detects the Table of Contents, and extracts individual items into structured JSON format with both HTML and plain text content.

## Features

- 🎯 **Smart Extraction**: Uses Table of Contents to accurately locate and extract specific items
- 📊 **Multiple Filing Types**: Supports both 10-K and 10-Q filings
- 🔄 **Batch Processing**: Extract from multiple companies, years, and filings in one command
- 🌐 **Full-Index Download**: Download ALL companies' filings when no ticker/CIK specified (uses SEC quarterly index files)
- 💾 **Skip Downloads**: Automatically skips re-downloading existing files
- � **CSV Reports**: Excel-ready CSV reports and real-time extraction logs with O/X success indicators
- 🎨 **Dual Format Output**: Each item saved as both HTML and plain text in JSON
- 🔍 **CIK or Ticker**: Works with both CIK numbers and stock ticker symbols
- ✓ **Amendment Filtering**: Automatically skips amended filings (10-K/A, 10-Q/A), selecting regular filings
- 🛡️ **Robust Boundary Detection**: Handles edge cases with ID-based markers and HTML parsing variations
- 📚 **Automatic Structure Extraction**: Detects and extracts nested heading-body pairs automatically during extraction
- 🔐 **Safety Prompts**: Confirmation required for large-scale downloads (thousands of filings)

## Installation

### Prerequisites

- Python 3.7 or higher
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ItemXtractor.git
cd ItemXtractor
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. **Important**: Update the User-Agent in `config.py`:
```python
SEC_USER_AGENT = "ItemXtractor/1.0 (Research Tool; your.email@domain.com)"
```
The SEC requires a valid email in the User-Agent header.

## Quick Start

### Command Line Usage

Extract all items from Apple's 2023 10-K:
```bash
python main.py --ticker AAPL --filing 10-K --year 2023
```

Extract specific items (Risk Factors and MD&A) from Microsoft's 2023 10-K:
```bash
python main.py --ticker MSFT --filing 10-K --year 2023 --items 1A 7
```

Extract from multiple companies and years:
```bash
python main.py --tickers AAPL MSFT GOOGL --filing 10-K --years 2022 2023
```

Download ALL companies for specific years (no ticker/CIK specified):
```bash
python main.py --filing 10-K --years 2023 2024 2025
```
⚠️ **Warning**: This will download thousands of filings and may take several hours.

### Peer Firm Analysis (peerfirm.py)

Identify similar peer companies using Gemini AI generative models. Two methods are available:

#### Method 1: Heading-Body Analysis (Local, No Indexing)
Uses the hierarchical structure from extracted items to generate peer recommendations via Gemini:

```bash
# Set API key (required)
$env:GEMINI_API_KEY="your-api-key-here"

# Find 5 peers based on Item 1 (Business) disclosure
python peerfirm.py --method headbody --k 5 --cik 0000001750 --year 2024 --item 1 --output ./output/peerfirm
```

This method:
- ✅ Reads `*_xtr.json` structure files (heading-body pairs)
- ✅ Constructs detailed business description from hierarchy
- ✅ Sends to Gemini for peer identification with reasoning
- ✅ No indexing required - fast for one-off queries
- ❌ Slower for batch analysis (one API call per company)

**Output files:**
- `{CIK}_{YEAR}_{ITEM}_{K}_headings_bodies_response.txt` - Formatted peer recommendations with reasoning

#### Method 2: Vector Database Similarity (Indexed, Fast Batch)
Builds embeddings for all companies and returns top-k similar companies:

```bash
# Build vector index for Item 1C (Cybersecurity)
python peerfirm_index.py --year 2024 --item 1C --filing 10-K --index-dir ./vector_db/peerfirm

# Query the index
python peerfirm.py --method vdb --k 5 --cik 0000001750 --year 2024 --item 1C --output ./output/peerfirm
```

**Build indexes for all items:**
```bash
# Build one index per item (e.g., 1, 1A, 1B, 1C, 7, ...)
python peerfirm_index.py --year 2024 --filing 10-K --index-dir ./vector_db/peerfirm
```

**Batch embeddings (reduce API calls):**
```bash
python peerfirm_index.py --year 2024 --item 1C --filing 10-K --index-dir ./vector_db/peerfirm --batch-size 20
```

Default batch size is `10`. If batch embedding fails, the script automatically falls back to single-item embedding calls.

**Index layout:**
```
vector_db/peerfirm/
├── item_1/
│   ├── embeddings.npy         # All company embeddings (N x 768)
│   ├── index.jsonl            # Company metadata (CIK, ticker, embedding index)
│   └── config.json            # Index metadata
├── item_1A/
│   ├── embeddings.npy
│   ├── index.jsonl
│   └── config.json
└── ...
```

**Output files (vdb method):**
- `{CIK}_{YEAR}_{ITEM}_{K}_vdb_matches.txt` - Top-k most similar companies by cosine distance

#### peerfirm.py Command Options

```bash
python peerfirm.py --method {headbody|vdb} --k 5 --cik CIK --year YEAR --item ITEM [--output DIR] [--save-prompt] [--keywords KEYWORDS]
```

**Arguments:**
- `--method {headbody|vdb}` - Analysis method (default: headbody)
- `--k` - Number of peers to return (default: 5)
- `--cik` - Company CIK (10-digit padded)
- `--year` - Filing year
- `--item` - Item number (1, 1A, 1B, 1C, 7, etc.)
- `--output` - Output directory (default: ./output/peerfirm)
- `--save-prompt` - Save the prompt sent to Gemini API (headbody only)
- `--keywords` - Custom keywords for peer search prompt (headbody only)

**Example with custom keywords:**
```bash
python peerfirm.py --method headbody --k 5 --cik 0000001750 --year 2024 --item 1 --output ./output/peerfirm --save-prompt --keywords "aircraft maintenance, defense contracting, supply chain"
```

**Environment variables:**
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` - **Required**. Get from [Google AI Studio](https://aistudio.google.com/app/apikey)
- `GEMINI_EMBED_MODEL` (default: `gemini-embedding-001`) - Embedding model for vdb method
- `GEMINI_GEN_MODEL` (default: `gemini-2-flash-preview`) - Generation model for headbody analysis
- `GEMINI_API_BASE` (default: `https://generativelanguage.googleapis.com/v1beta`)
- `GEMINI_MAX_RETRIES` (default: `5`)
- `GEMINI_BACKOFF_BASE` (default: `2.0` seconds) - Exponential backoff base for rate limiting

**Set environment variable (Windows PowerShell):**
```powershell
$env:GEMINI_API_KEY="your-api-key-here"
python peerfirm.py --method headbody --k 5 --cik 0000001750 --year 2024 --item 1 --output ./output/peerfirm
```

#### Choosing a Method

| Aspect | headbody | vdb |
|--------|----------|-----|
| **Setup** | None (use directly) | Requires indexing |
| **Speed (single query)** | Slower (1 API call) | Instant |
| **Speed (batch)** | Very slow (N API calls) | Fast (all at once) |
| **Quality** | Generative (reasons) | Similarity-based |
| **Cost** | Higher (API calls/query) | Lower (API calls upfront) |
| **Best for** | Few one-off queries | Many queries, same year/item |

**Recommendations:**
- Use `headbody` for initial exploration and detailed reasoning
- Use `vdb` for batch analysis across multiple companies/years
- Combine both: use `vdb` to find candidates, then `headbody` for detailed analysis

#### TODO
- Add prompt/response evaluation metrics (e.g., agreement with known peer sets)
- Add optional candidate filtering by SIC/NAICS before similarity ranking
- Compare `--method headbody` vs `vdb` results on benchmark set

### Python API Usage

```python
from main import ItemXtractor

# Create extractor instance
extractor = ItemXtractor()

# Extract all items from a filing
extractor.extract(
    cik_tickers="AAPL",
    filing_types="10-K",
    years=2023,
    items=None  # None = extract all items
)

# Extract specific items
extractor.extract(
    cik_tickers=["AAPL", "MSFT"],
    filing_types="10-K",
    years=[2022, 2023],
    items=["1", "1A", "7"]  # Business, Risk Factors, MD&A
)
```

## File Structure

Extracted filings and items are organized by **CIK number**. When using `--ticker`, the tool automatically converts it to CIK for consistent folder naming:

```
sec_filings/
├── 0000320193/              # AAPL CIK
│   └── 2023/
│       └── 10-K/
│           ├── 0000320193_2023_10-K.html    # Original filing
│           └── items/
│               ├── 0000320193_2023_10-K_item1.json
│               ├── 0000320193_2023_10-K_item1_xtr.json
│               ├── 0000320193_2023_10-K_item1A.json
│               ├── 0000320193_2023_10-K_item1A_xtr.json
│               └── ...
└── 0000789019/              # MSFT CIK
    └── 2023/
        └── 10-Q/
            ├── 0000789019_2023_10-Q.html
            └── items/
                └── ...
```

**Why CIK-based naming?**
- ✅ Consistent with SEC EDGAR structure
- ✅ Prevents duplicate downloads (ticker vs CIK references)
- ✅ Handles ticker changes over time (CIKs are permanent)
- ✅ Works seamlessly with full-index downloads

### Structure Extraction

Structure extraction happens **automatically** during item extraction. Each extracted item gets a corresponding `*_xtr.json` file containing hierarchical heading-body pairs.

**Example extraction:**
```bash
python main.py --ticker AAPL --filing 10-K --year 2022
```

**Output files (in `sec_filings/0000320193/2022/10-K/items/`):**
- `0000320193_2022_10-K_item1.json` - Item content (HTML + text)
- `0000320193_2022_10-K_item1_xtr.json` - Hierarchical structure (automatically created)

**Structure file format:**

```json
{
  "ticker": "AAPL",
  "year": "2022",
  "filing_type": "10-K",
  "item_number": "1",
  "structure": [
    {
      "type": "bold_heading",
      "layer": 1,
      "heading": "Products",
      "body": "",
      "children": [
        {
          "type": "heading",
          "layer": 2,
          "heading": "iPhone",
          "body": "iPhone® is the Company's line of smartphones...",
          "children": []
        },
        {
          "type": "heading",
          "layer": 2,
          "heading": "Mac",
          "body": "Mac® is the Company's line of personal computers...",
          "children": []
        }
      ]
    }
  ]
}
```

**How it works:**
- Level 1 headings are bold styled divs (font-weight:700)
- Level 2 headings are italic styled divs (font-style:italic) nested under level 1
- Each heading captures the body content until the next heading at same or higher level
- Supports arbitrary nesting depth for complex documents

**Example - AAPL 2022 Item 1 structure:**
- Item 1. Business (level 1)
  - Company Background (level 1)
  - Products (level 1)
    - iPhone (level 2)
    - Mac (level 2)
    - iPad (level 2)
    - Wearables, Home and Accessories (level 2)
  - Services (level 1)
    - Advertising (level 2)
    - AppleCare (level 2)
    - Cloud Services (level 2)
    - Digital Content (level 2)
    - Payment Services (level 2)
  - Human Capital (level 1)
    - Workplace Practices and Policies (level 2)
    - Compensation and Benefits (level 2)
    - Inclusion and Diversity (level 2)
    - Engagement (level 2)
    - Health and Safety (level 2)

```

## Available Items

### 10-K Filing Items

| Item | Description |
|------|-------------|
| 1    | Business |
| 1A   | Risk Factors |
| 1B   | Unresolved Staff Comments |
| 1C   | Cybersecurity |
| 2    | Properties |
| 3    | Legal Proceedings |
| 4    | Mine Safety Disclosures |
| 5    | Market for Registrant's Common Equity |
| 6    | Selected Financial Data (removed in newer filings) |
| 7    | Management's Discussion and Analysis |
| 7A   | Quantitative and Qualitative Disclosures About Market Risk |
| 8    | Financial Statements and Supplementary Data |
| 9    | Changes in and Disagreements with Accountants |
| 9A   | Controls and Procedures |
| 9B   | Other Information |
| 10   | Directors, Executive Officers and Corporate Governance |
| 11   | Executive Compensation |
| 12   | Security Ownership of Certain Beneficial Owners and Management |
| 13   | Certain Relationships and Related Transactions |
| 14   | Principal Accounting Fees and Services |
| 15   | Exhibits, Financial Statement Schedules |
| 16   | Form 10-K Summary |

### 10-Q Filing Items

| Item | Description |
|------|-------------|
| 1    | Financial Statements |
| 2    | Management's Discussion and Analysis |
| 3    | Quantitative and Qualitative Disclosures About Market Risk |
| 4    | Controls and Procedures |

## Full-Index Download (All Companies)

When no `--ticker` or `--cik` is specified, ItemXtractor automatically downloads filings for **ALL companies** from SEC EDGAR using quarterly full-index files.

### How It Works

1. Downloads `company.idx` files from https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{1-4}/
2. Parses index to extract all companies that filed the specified form type
3. Removes duplicates and prompts for confirmation
4. Downloads and extracts items from all discovered filings

### Usage

```bash
# Download ALL 10-K filings for 2024 (will prompt for confirmation)
python main.py --filing 10-K --year 2024

# Download ALL companies across multiple years
python main.py --filing 10-K --years 2023 2024 2025

# Download all companies but extract specific items only
python main.py --filing 10-K --year 2024 --items 1 1A 7
```

### Expected Volume & Time

**Typical 10-K volume per year:** ~6,700 companies  
**Typical 10-Q volume per year:** ~20,000 filings (3 quarters)

**Time estimates for 6,700 10-K filings:**
- Index download: ~1.6 seconds
- Filing downloads: ~22 minutes minimum (with SEC rate limiting)
- Full extraction: 2-3 hours

**Storage:** 3-13 GB per 6,700 filings

### Safety Features

- **Confirmation Prompt**: Shows estimated filing count and requires "yes" to proceed
- **Progress Reporting**: Displays companies found per filing type
- **Rate Limiting**: Respects SEC's 10 requests/second limit
- **Graceful Errors**: Continues processing if individual filings fail

Example confirmation prompt:
```
================================================================================
WARNING: No tickers or CIKs specified - will download ALL companies!
================================================================================

Estimated 10-K filings: ~6,700
Years: 2024
Quarters to check: 4

This may take several hours and require significant storage.
Rate limiting: 10 requests/second (SEC requirement)

Do you want to continue? (yes/no):
```

### Best Practices

1. **Test with single year first** before scaling to multiple years
2. **Use `--items`** to limit extraction scope and reduce processing time
3. **Monitor disk space** before starting large downloads
4. **Use high `--workers`** count for parallel processing (e.g., `--workers 16`)

See [FULL_INDEX_FEATURE.md](FULL_INDEX_FEATURE.md) for detailed documentation.

## CSV Reports and Logs

ItemXtractor generates **Excel-ready CSV files** for easy analysis and monitoring:

### Extraction Log (Real-time)
- **File**: `logs/extraction_{timestamp}.csv`
- **Purpose**: Real-time progress tracking as filings complete
- **Updates**: Row added immediately after each filing completes
- **Format**: O/X matrix with Download, TOC, and all items

### Final Report
- **File**: `logs/report_{timestamp}.csv`
- **Purpose**: Complete summary generated at the end of extraction
- **Includes**: Success/failure statistics and total runtime
- **Format**: Same O/X matrix as extraction log + summary section

### CSV Structure

```csv
Start Time: 2024-01-15 10:30:45

Ticker,Year,Filing Type,Download,TOC,Item 1,Item 1A,Item 1B,Item 2,...,Runtime (s)
AAPL,2023,10-K,O,O,O,O,O,O,...,45.23
MSFT,2023,10-K,O,O,O,O,X,O,...,52.18
GOOGL,2023,10-K,O,X,,,,,,...,12.45

Summary:
Total Filings Attempted: 3
Successful Downloads: 3
Failed Downloads: 0
Total Items Extracted: 45
Failed Items: 1
Total Runtime: 109.86 seconds
```

### Indicator Legend
- **O**: Success (item extracted/downloaded)
- **X**: Failure (error occurred)
- **Empty**: N/A (item skipped or not applicable)

### Opening in Excel
Both CSV files can be opened directly in Excel, Google Sheets, or analyzed with Pandas:

```python
import pandas as pd

# Read extraction log
df = pd.read_csv('logs/extraction_20240115_103045.csv', skiprows=1)

# Filter failures
failures = df[(df == 'X').any(axis=1)]
```

## Filing Statistics Report (stat.py)

Generate a **comprehensive descriptive statistics report** from downloaded filings and extracted structures:

```bash
python stat.py --folder sec_filings
```

**Output:**
- `stats/filing_analysis_{timestamp}.md`

**What it includes:**
- Item extraction counts by year (sorted by `ITEMS_10K`)
- Structure depth metrics by year
- Headings & bodies statistics by item and year
- Extra/unknown item investigation with CIK + TOC titles
- Error report summary by year

**Notes:**
- Uses TOC-derived `item_title` from extracted item JSON
- Groups sub-items (e.g., `8A`, `8B`) under their parent item (`8`)
- Designed for large datasets (no HTML parsing required)

## Command Line Options

```
usage: main.py [-h] [--ticker TICKERS [TICKERS ...]] [--cik CIKS [CIKS ...]]
               --filing {10-K,10-Q} [{10-K,10-Q} ...]
               --year YEARS [YEARS ...]
               [--items ITEMS [ITEMS ...]]
               [--output-dir OUTPUT_DIR]
               [--log-dir LOG_DIR]
               [--workers WORKERS]

Extract items from SEC EDGAR filings

optional arguments:
  -h, --help            show this help message and exit
  --ticker TICKERS [TICKERS ...], --tickers TICKERS [TICKERS ...]
                        Stock ticker symbol(s) (optional - omit for all companies)
  --cik CIKS [CIKS ...], --ciks CIKS [CIKS ...]
                        CIK number(s) (optional - omit for all companies)
  --filing {10-K,10-Q} [{10-K,10-Q} ...], --filings {10-K,10-Q} [{10-K,10-Q} ...]
                        Filing type(s)
  --year YEARS [YEARS ...], --years YEARS [YEARS ...]
                        Year(s) to extract
  --items ITEMS [ITEMS ...]
                        Item number(s) to extract (default: all)
  --output-dir OUTPUT_DIR
                        Output directory for filings (default: sec_filings)
  --log-dir LOG_DIR     Log directory (default: logs)
  --workers WORKERS     Number of parallel workers (default: 4)
```

## Logging and Reports

ItemXtractor generates comprehensive logs for every extraction session:

- **Console logs**: Real-time progress output
- **Log files**: Detailed logs saved in `logs/extraction_YYYYMMDD_HHMMSS.log`
- **JSON reports**: Summary reports saved in `logs/report_YYYYMMDD_HHMMSS.json`

### Report Contents

Each JSON report includes:
- Parameters used for extraction
- Start and end timestamps
- Total duration
- For each filing:
  - Download status (new download or skipped)
  - TOC detection results
  - Successfully extracted items
  - Any errors encountered
  - Processing time

## Usage Examples

For programmatic use, import and instantiate the `ItemXtractor` class:

```python
from main import ItemXtractor

extractor = ItemXtractor()
extractor.extract(
    cik_tickers="AAPL",
    filing_types="10-K",
    years=2022,
    items=["1", "1A", "7"]
)
```

## How It Works

1. **Resolution**: Converts ticker symbols to CIK numbers using SEC's company tickers API
2. **Download**: Fetches the filing HTML from SEC EDGAR (or skips if already downloaded)
   - **Amendment Filtering**: Checks document types and skips amended filings (10-K/A, 10-Q/A)
3. **TOC Detection**: Intelligently locates the Table of Contents in the filing
4. **Parsing**: Extracts anchor links and positions for each item from the TOC
5. **Extraction**: Uses TOC information to accurately split the filing into individual items
   - **ID-Based Boundary Detection**: Uses HTML element IDs (signatures, exhibits, cover) for precise item boundaries
   - **Fallback Patterns**: Handles cases where text spans multiple HTML tags
6. **Conversion**: Generates both HTML and plain text versions of each item
7. **Storage**: Saves each item as a structured JSON file
8. **Logging**: Records all activities and generates a comprehensive report

## Requirements

- `requests>=2.31.0` - HTTP requests to SEC EDGAR
- `beautifulsoup4>=4.12.0` - HTML parsing
- `lxml>=4.9.0` - Fast XML/HTML parsing
- `html5lib>=1.1` - HTML5 parsing support
- `numpy>=1.24.0` - Vector math for peer-firm similarity

## SEC API Guidelines

This tool follows SEC EDGAR's API guidelines:
- Maximum 10 requests per second (configurable in `config.py`)
- Declares a User-Agent header with contact information
- Respects robots.txt

**Please update the User-Agent with your email before using this tool.**

## Limitations

- **TOC Dependency**: If a filing doesn't have a detectable Table of Contents, items cannot be extracted. The tool will log this and skip extraction.
- **Format Variations**: SEC filings vary in format. While the tool handles most common formats, some unusual formats may not parse correctly.
- **Historical Filings**: Very old filings may have different structures. The tool is optimized for recent filings (2010+).

## Troubleshooting

### Amendment Filings
- The tool automatically filters out amended filings (10-K/A, 10-Q/A) and selects the original filing type
- This ensures you get the current, non-amended version of the filing

### Item Boundaries and Signatures Section
- The tool uses ID-based markers in HTML elements to detect item boundaries precisely
- This handles edge cases where text (like "SIGNATURES") is split across multiple HTML tags
- If Item 16 is extracted, it correctly stops before the SIGNATURES section

### No TOC Found

If the tool reports "No TOC found":
- The filing may use an unusual format
- Try manually inspecting the HTML file in `sec_filings/`
- Some filings don't have a traditional TOC

### Item Not Extracted

If specific items aren't extracted:
- Check the log file for errors
- Verify the item exists in that filing type (10-K vs 10-Q have different items)
- The TOC may not include that item number

### Download Failures

If downloads fail:
- Check your internet connection
- Verify the ticker/CIK is correct
- Ensure you've updated the User-Agent with a valid email
- The filing may not exist for that year

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is provided as-is for research purposes. Please ensure compliance with SEC EDGAR's terms of service and applicable data usage policies.

## Acknowledgments

- SEC EDGAR for providing free access to financial filings
- Beautiful Soup for HTML parsing capabilities

## Contact

For issues, questions, or contributions, please use the GitHub issue tracker.

---

**Disclaimer**: This tool is for research and educational purposes. Always verify extracted data against original SEC filings. The authors are not responsible for any decisions made based on data extracted using this tool.
