# Performance Analysis: Item Extraction Bottlenecks

## Current Architecture Flow
```
HTML Download → Parse TOC → Get Item Positions → Extract Each Item → Convert to Text
```

## Identified Performance Bottlenecks

### 1. **TEXT CONVERSION (BIGGEST BOTTLENECK)** ⚠️⚠️⚠️
**Location:** `src/extractor.py` - `_html_to_text()` method  
**Issue:** Called ONCE PER ITEM, processes ENTIRE item HTML  
**Cost:** Very expensive for large items

**Current Process:**
```python
def _html_to_text(self, html_content: str) -> str:
    # 1. BeautifulSoup parsing (high memory/CPU)
    soup = BeautifulSoup(html_with_breaks, 'html.parser')
    
    # 2. Remove scripts/styles (traverses entire DOM tree)
    for script in soup(['script', 'style']):
        script.decompose()
    
    # 3. Extract text with separators (traverses entire DOM again)
    text = soup.get_text(separator=' ')
    
    # 4. Cleanup & strip headers/footers (complex regex operations)
    text = self._strip_headers_footers(text)
```

**Problems:**
- BeautifulSoup re-parses already-parsed HTML
- `soup.get_text()` traverses entire DOM tree for every item
- `_strip_headers_footers()` uses word-by-word artifact detection with multiple regex searches
- For large items (Item 8 with 500+ pages = 5-10MB HTML), this is very slow

**Impact on 30 companies × 5 years:**
- 150 filings × ~15 items per filing = 2,250 items
- At ~2-5 seconds per large item = **5,000-12,500 seconds = 1.4-3.5 hours just on text conversion**

---

### 2. **REDUNDANT HTML PARSING**
**Location:** `src/extractor.py` - `extract_item()` method  
**Issue:** Parses HTML twice per item

**Current Code:**
```python
def extract_item(self, html_content, item_number, toc_items):
    # ...
    item_html = html_content[start_pos:end_pos]  # Slice HTML string
    
    # Parse 1st time: _clean_html
    item_html_clean = self._clean_html(item_html)
    
    # Parse 2nd time: _html_to_text (creates new BeautifulSoup)
    item_text = self._html_to_text(item_html)
```

**Problems:**
- Each item HTML is parsed twice by BeautifulSoup
- First parse just cleans it, second parse extracts text
- Could be combined into single parse

---

### 3. **REGEX SEARCH ON ENTIRE FILING (Linear Scan)**
**Location:** `src/parser.py` - `get_item_positions()` method  
**Issue:** Multiple regex searches on full HTML for each item

**Current Code:**
```python
for item_num in sorted_items:
    # Each iteration searches entire html_content from start position
    anchor_match = re.search(anchor_pattern, html_content, re.IGNORECASE)
    
    # Then searches again from start_pos
    next_anchor_match = re.search(pattern, html_content[start_pos:])
    
    # For last item, searches 5 different patterns on entire document
    for marker_pattern in end_markers:
        marker_match = re.search(marker_pattern, html_content[start_pos:])
```

**Problems:**
- 5+ regex searches per item in sequence
- Each regex search scans large portions of HTML
- No caching or pre-compilation
- For last item, tries 5 patterns sequentially (worst case: 5× search cost)

---

### 4. **HEADER/FOOTER STRIPPING (Expensive Artifact Detection)**
**Location:** `src/extractor.py` - `_strip_headers_footers()` method  
**Issue:** Complex multi-level word checking for every page

**Current Code:**
```python
while words:
    # Check last word
    if is_artifact_phrase(words[-1]):
        words.pop()
    # Check last 2 words
    elif len(words) >= 2 and is_artifact_phrase(' '.join(words[-2:])):
        words.pop()
        words.pop()
    # Check last 3 words
    elif len(words) >= 3 and is_artifact_phrase(' '.join(words[-3:])):
        # ... repeat for 4 and 5 words
```

**Problems:**
- O(n) algorithm that checks multiple sub-phrases
- Joins strings repeatedly (creates new strings each check)
- Checks both ends of document (beginning + end)
- For items with 10,000+ words, this is slow
- regex patterns in `is_artifact_phrase()` called hundreds of times

**Example:** Item 8 (Financial Statements) with 100+ pages:
- 50,000+ words total
- Could need 5,000+ phrase checks
- Expensive regex operations

---

### 5. **SEQUENTIAL ITEM EXTRACTION**
**Location:** `main.py` - `process_filing()` method  
**Issue:** Processes items one at a time in a loop

**Current Code:**
```python
for item_number in items_to_extract:
    try:
        item_data = self.extractor.extract_item(html_content, item_number, toc_items)
        # Save, log, etc.
```

**Problems:**
- No parallelization (Python GIL not an issue here since mostly I/O)
- Could extract 15 items in parallel, but instead does sequentially
- For 30 companies × 15 items = 450 sequential extractions

---

## Performance Ranking (Worst to Best)

| Rank | Bottleneck | Estimated Time | % of Total |
|------|-----------|-----------------|-----------|
| 1️⃣  | Text Conversion (`_html_to_text`) | 1.4-3.5 hrs | 60-70% |
| 2️⃣  | Regex Searches (`get_item_positions`) | 15-30 min | 5-10% |
| 3️⃣  | Header/Footer Stripping (`_strip_headers_footers`) | 20-40 min | 5-10% |
| 4️⃣  | HTML Parsing/Cleaning (redundant) | 10-20 min | 3-5% |
| 5️⃣  | Sequential Extraction (no parallelization) | 5-15 min | 1-3% |
| 6️⃣  | Other (logging, I/O, TOC parsing) | 10-20 min | 3-5% |

---

## Optimization Opportunities

### ✅ HIGH IMPACT (Quick Wins)
1. **Combine HTML parsing** (2x speedup)
   - Parse once, extract text and cleaned HTML from same parse
   
2. **Cache regex patterns** (1.5x speedup)
   - Pre-compile regex patterns instead of compiling each time
   
3. **Simplify artifact detection** (1.5-2x speedup)
   - Remove nested word-checking, use simpler heuristic
   - Pre-compute artifact patterns

4. **Use lxml parser instead of html.parser** (1.5-2x speedup)
   - lxml is much faster than html.parser
   - Minimal code changes

### 🚀 MEDIUM IMPACT
5. **Parallel item extraction** (up to 10-15x for I/O-bound work)
   - Extract multiple items concurrently
   - ThreadPoolExecutor or multiprocessing
   
6. **Stream-based text extraction**
   - Don't load entire HTML in memory, process chunks
   - Reduces memory pressure

### 📊 LOW IMPACT (Complex, Minor Benefit)
7. **Index item positions** 
   - Pre-compute item positions in dictionary
   - Only helps if extracting same file multiple times

---

## Recommended Implementation Priority

### Phase 1: Quick Wins (Target: 3-4x speedup, 1-2 hours work)
1. Use lxml parser
2. Pre-compile regex patterns
3. Combine parsing operations
4. Simplify artifact detection

### Phase 2: Parallel Extraction (Target: 10-15x additional speedup, 1-2 hours work)
1. Implement ThreadPoolExecutor
2. Extract multiple items concurrently
3. Maintain thread-safe logging

### Phase 3: Advanced Optimization (Target: 1.5-2x additional speedup, 3-4 hours work)
1. Stream-based processing for huge files
2. Lazy BeautifulSoup parsing
3. Better artifact detection heuristics

---

## Estimated Current Performance
**Scenario:** 30 companies × 5 years × 15 items = 2,250 items

**Breakdown:**
- Text conversion: 60-70% of time = 1.4-3.5 hours
- Regex/Artifact detection: 15-20% = 20-30 min
- Other operations: 10-15% = 10-20 min

**Total: 2-4 hours for full extraction**

---

## After Phase 1 Optimizations (Expected)
**Target:** 30-40 minutes (4-5x improvement)

## After Phase 1 + Phase 2 Optimizations (Expected)
**Target:** 3-5 minutes (40-50x improvement)
