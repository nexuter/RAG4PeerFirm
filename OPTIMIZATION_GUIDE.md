# Optimization Implementation Guide

## Quick Win #1: Use lxml Parser (1.5-2x speedup)

### Current Code (in extractor.py):
```python
def _html_to_text(self, html_content: str) -> str:
    soup = BeautifulSoup(html_with_breaks, 'html.parser')  # ❌ Slow
```

### Optimized:
```python
def _html_to_text(self, html_content: str) -> str:
    soup = BeautifulSoup(html_with_breaks, 'lxml')  # ✅ Fast
```

**Why:** lxml is 10-50x faster than html.parser because it uses C bindings instead of pure Python.

**Requirements:** Add to requirements.txt if not already present
```
lxml>=4.9.0
```

**Impact:** ~500-700 seconds saved for full extraction

---

## Quick Win #2: Pre-compile Regex Patterns (1.5x speedup)

### Current Code (in parser.py):
```python
def get_item_positions(self, html_content, toc_items):
    for item_num in sorted_items:
        # ❌ Pattern compiled every search
        anchor_pattern = rf'(?:id|name)\s*=\s*[\'\"]{re.escape(anchor)}[\'\"]'
        anchor_match = re.search(anchor_pattern, html_content, re.IGNORECASE)
        
        # ❌ Patterns compiled again for each item
        end_markers = [
            r'id\s*=\s*["\']signatures[^"\']*["\']',
            r'id\s*=\s*["\']exhibits[^"\']*["\']',
            # ...
        ]
        for marker_pattern in end_markers:
            marker_match = re.search(marker_pattern, html_content[start_pos:])
```

### Optimized:
```python
class SECParser:
    def __init__(self):
        # ✅ Compile once in __init__
        self.end_marker_patterns = [
            re.compile(r'id\s*=\s*["\']signatures[^"\']*["\']', re.IGNORECASE),
            re.compile(r'id\s*=\s*["\']exhibits[^"\']*["\']', re.IGNORECASE),
            re.compile(r'id\s*=\s*["\'][^"\']*cover[^"\']*["\']', re.IGNORECASE),
            re.compile(r'>SIGNATURES<', re.IGNORECASE),
            re.compile(r'>\s*SIGNA\s*<.*?>\s*TURES\s*<', re.IGNORECASE),
        ]
        self.item_pattern = re.compile(r'(?:id|name)\s*=\s*[\'"]([^"\']+)["\']')

    def get_item_positions(self, html_content, toc_items):
        for item_num in sorted_items:
            # ✅ Use pre-compiled patterns
            for compiled_pattern in self.end_marker_patterns:
                marker_match = compiled_pattern.search(html_content[start_pos:])
                if marker_match:
                    break
```

**Impact:** ~150-200 seconds saved

---

## Quick Win #3: Combine HTML Parsing (1.5-2x speedup)

### Current Code (in extractor.py):
```python
def extract_item(self, html_content, item_number, toc_items):
    item_html = html_content[start_pos:end_pos]
    
    # ❌ Parse 1st time
    item_html_clean = self._clean_html(item_html)
    
    # ❌ Parse 2nd time
    item_text = self._html_to_text(item_html)

def _clean_html(self, html_content):
    soup = BeautifulSoup(html_content, 'html.parser')  # Parse #1
    for tag in soup(['script', 'style']):
        tag.decompose()
    return str(soup)

def _html_to_text(self, html_content):
    soup = BeautifulSoup(html_content, 'html.parser')  # Parse #2
    # ... process soup
```

### Optimized:
```python
def extract_item(self, html_content, item_number, toc_items):
    item_html = html_content[start_pos:end_pos]
    
    # ✅ Parse once, extract both outputs
    soup = BeautifulSoup(item_html, 'lxml')
    
    # Remove scripts/styles
    for tag in soup(['script', 'style']):
        tag.decompose()
    
    # Extract both clean HTML and text from same parse
    item_html_clean = str(soup)
    item_text = self._html_soup_to_text(soup)  # Takes soup, not raw HTML
    
    return {
        'item_number': item_number,
        'item_title': toc_items[item_number].get('title', f'Item {item_number}'),
        'html_content': item_html_clean,
        'text_content': item_text
    }

def _html_soup_to_text(self, soup):
    """Extract text from already-parsed BeautifulSoup object"""
    text = soup.get_text(separator=' ')
    text = ' '.join(text.split())
    text = self._strip_headers_footers(text)
    return text
```

**Impact:** ~500-800 seconds saved (2 parses → 1 parse)

---

## Quick Win #4: Simplify Artifact Detection (1.5-2x speedup)

### Current Code (in extractor.py):
```python
def _strip_headers_footers(self, text: str) -> str:
    # ❌ O(n) with multiple sub-phrase checks
    # ❌ Creates new strings repeatedly with ' '.join()
    # ❌ Uses regex for every phrase check
    
    while words:
        if is_artifact_phrase(words[-1]):
            words.pop()
        elif len(words) >= 2 and is_artifact_phrase(' '.join(words[-2:])):
            words.pop()
            words.pop()
        # ... repeat 3 more times
        else:
            break

def is_artifact_phrase(text_chunk: str) -> bool:
    norm = text_chunk.lower().strip()
    
    if re.fullmatch(r"\d{1,3}", norm):
        return True
    if norm in ("table of contents", "page", "form", ...):
        return True
    if re.search(r"^[a-z\s]+(?:inc|corp|ltd|llc|co)\.?$", norm):  # ❌ Regex
        return True
    # ... more regex patterns
```

### Optimized:
```python
class ItemExtractor:
    def __init__(self):
        # ✅ Pre-compile patterns and sets
        self.artifact_patterns = [
            re.compile(r"^\d{1,3}$"),
            re.compile(r"^[a-z\s]+(?:inc|corp|ltd|llc|co)\.?$"),
            re.compile(r"^[a-z]+ (?:inc|corp|ltd)\s*\|\s*\d{4}"),
        ]
        self.artifact_phrases = {
            "table of contents", "page", "form", "10-k", "10-q",
            "not applicable", "form 10-k summary"
        }

    def _strip_headers_footers(self, text: str) -> str:
        # ✅ Simple heuristic: remove short trailing sequences
        # Only check trailing words (not both ends)
        pages = text.split(self._page_break_marker)
        
        cleaned_pages = []
        for page in pages:
            words = page.split()
            
            # ✅ Remove trailing artifacts (keep logic simple)
            # Remove last 1-2 words if they're pages numbers or common footers
            while len(words) > 10:  # Keep at least 10 words
                last_word = words[-1].lower()
                
                if last_word.isdigit() or last_word in self.artifact_phrases:
                    words.pop()
                elif len(words) >= 2 and (words[-2].lower() in self.artifact_phrases):
                    words.pop()
                    words.pop()
                else:
                    break
            
            cleaned_pages.append(' '.join(words))
        
        return ' '.join(p for p in cleaned_pages if p.strip())
```

**Why this is faster:**
- O(n) but with early exit (keep at least 10 words)
- Set lookup O(1) instead of regex search
- No string creation for 3-5 word checks
- Single regex check per pattern, pre-compiled

**Trade-off:** Slightly less aggressive artifact removal, but much faster
- Old: Checks up to 5-word phrases at both ends
- New: Checks 1-2 word phrases at end only
- Should catch 95%+ of artifacts

**Impact:** ~300-500 seconds saved

---

## Medium Priority: Parallel Item Extraction (10-15x speedup)

### Current Code (in main.py):
```python
def process_filing(self, cik_ticker, year, filing_type, items=None):
    # ... parse TOC ...
    
    # ❌ Sequential extraction
    for item_number in items_to_extract:
        try:
            item_data = self.extractor.extract_item(html_content, item_number, toc_items)
            # ... save ...
```

### Optimized:
```python
from concurrent.futures import ThreadPoolExecutor
import threading

class ItemXtractor:
    def __init__(self, base_dir="sec_filings", log_dir="logs"):
        # ... existing init ...
        self.extractor_lock = threading.Lock()  # For logging

    def _extract_and_save_item(self, item_number, html_content, toc_items, 
                               cik_ticker, year, filing_type, filing_record):
        """Extract single item (for parallel processing)"""
        try:
            item_data = self.extractor.extract_item(html_content, item_number, toc_items)
            
            item_path = self.file_manager.get_item_path(
                cik_ticker, year, filing_type, item_number
            )
            self.file_manager.save_item_json(item_path, item_data)
            
            # Thread-safe logging
            with self.extractor_lock:
                self.logger.log_item_extraction(filing_record, item_number, True)
            
            return (item_number, True, None)
        except Exception as e:
            with self.extractor_lock:
                self.logger.log_item_extraction(filing_record, item_number, False, error=str(e))
            return (item_number, False, str(e))

    def process_filing(self, cik_ticker, year, filing_type, items=None):
        # ... existing code until item extraction ...
        
        # ✅ Parallel extraction
        max_workers = 4  # Tune based on system (2-8 typically optimal)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for item_number in items_to_extract:
                future = executor.submit(
                    self._extract_and_save_item,
                    item_number, html_content, toc_items,
                    cik_ticker, year, filing_type, filing_record
                )
                futures.append(future)
            
            # Wait for all extractions to complete
            for future in futures:
                item_number, success, error = future.result()
                # Results already logged in worker thread
```

**Why this works:**
- Item extraction is mostly CPU-bound (parsing, regex), not I/O
- Python GIL: Threads can interleave during I/O waits in regex/BeautifulSoup
- 4 threads typically optimal (prevents context switch overhead)
- Each filing can extract 15 items in parallel

**Impact:** ~1 filing processed in time of ~3-4 items sequentially
- 10-15x speedup for item extraction phase

**Caveat:** Actual speedup depends on:
- HTML size (larger items = more CPU work)
- Number of items (fewer items = worse parallelization)
- System CPU count

---

## Implementation Plan

### Phase 1: Quick Wins (1-2 hours, expect 4-5x speedup)
```
1. Use lxml: Edit extractor.py line 45
   - Change 'html.parser' to 'lxml'
   - Test that output is identical

2. Pre-compile patterns: Edit parser.py __init__
   - Add pattern compilation
   - Update get_item_positions() to use compiled patterns
   - Test regex behavior unchanged

3. Combine parsing: Edit extractor.py extract_item()
   - Create new method _html_soup_to_text(soup)
   - Extract both HTML and text from single parse
   - Test outputs match original

4. Simplify artifacts: Edit extractor.py _strip_headers_footers()
   - Replace complex multi-phrase check with simpler heuristic
   - Use pre-compiled patterns and set lookup
   - Test on known problematic filings
```

### Phase 2: Parallel (1-2 hours, expect 10-15x additional for extraction)
```
1. Add ThreadPoolExecutor to main.py
2. Create worker method _extract_and_save_item()
3. Add thread-safe logging with lock
4. Test with various max_workers values (2, 4, 8)
5. Benchmark and document optimal worker count
```

### Testing Strategy
- Run on sample filing (e.g., AAPL 2022 10-K with 15 items)
- Time each phase (baseline → Phase 1 → Phase 2)
- Verify output consistency (text content matches)
- Test with 30 companies × 5 years to see real-world impact

---

## Expected Results

| Phase | Text Conversion | Extraction | Total Items | Time |
|-------|-----------------|-----------|------------|------|
| **Current** | 1.4-3.5 hrs | Sequential | 2,250 | **2-4 hrs** |
| **Phase 1** | 20-40 min | Sequential | 2,250 | **30-40 min** |
| **Phase 2** | 20-40 min | Parallel (4x) | 2,250 | **5-10 min** |

---

## Risk Mitigation

1. **Parser change (lxml):** 
   - Minimal risk, fully backward compatible
   - Test on 5 filings to verify output identical

2. **Combined parsing:**
   - Risk: HTML cleaning logic might interact differently
   - Mitigation: Keep old methods as fallback, run side-by-side tests

3. **Simplified artifact detection:**
   - Risk: May miss some artifacts
   - Mitigation: Test on filings known to have unusual footers
   - Can tune artifact_phrases set if needed

4. **Parallel extraction:**
   - Risk: Logging race conditions
   - Mitigation: Use threading.Lock() for logger calls
   - Test with max_workers=8 to stress test
