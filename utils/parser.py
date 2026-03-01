"""
SEC Filing Parser - Detects and parses Table of Contents
"""

import re
import unicodedata
from typing import Dict, Optional, Tuple
from bs4 import BeautifulSoup, Tag


class SECParser:
    """Parses SEC filings to extract Table of Contents"""
    
    def __init__(self):
        """Initialize SECParser with pre-compiled regex patterns"""
        # OPTIMIZED: Pre-compile patterns for reuse
        self.end_marker_patterns = [
            re.compile(r'id\s*=\s*["\']signatures[^"\']*["\']', re.IGNORECASE),
            re.compile(r'id\s*=\s*["\']exhibits[^"\']*["\']', re.IGNORECASE),
            re.compile(r'id\s*=\s*["\'][^"\']*cover[^"\']*["\']', re.IGNORECASE),
            re.compile(r'>SIGNATURES<', re.IGNORECASE),
            re.compile(r'>\s*SIGNA\s*<.*?>\s*TURES\s*<', re.IGNORECASE),
        ]
        self.item_pattern = re.compile(r'item\s+(\d+[A-Za-z]?)\b', re.IGNORECASE)
        self.part_item_pattern = re.compile(r'part\s+[IV]+\s*[–-]\s*item\s+(\d+[A-Za-z]?)\b', re.IGNORECASE)
        self.part_heading_html_pattern = re.compile(
            r'>\s*PART(?:\s|&nbsp;)+[IVXLC]+\b',
            re.IGNORECASE,
        )
    
        self.toc_marker_pattern = re.compile(
            r'table\s+of\s+contents|index\s+to\s+financial\s+statements',
            re.IGNORECASE,
        )
        # TOC should appear near the beginning of the filing.
        # Inline XBRL filings can prepend very large hidden headers, so we
        # allow a larger offset while still requiring explicit TOC markers.
        self.max_toc_marker_offset = 4000000
        self.toc_region_padding_before = 3000
        self.toc_region_length = 260000
        # Fallback window when no explicit TOC marker exists.
        self.toc_fallback_prefix_length = 800000

    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing extra whitespace
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Normalize and remove invisible formatting chars (generic cleanup)
        text = unicodedata.normalize("NFKC", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
        text = re.sub(r"[\u2018\u2019\u201A\u201B\u2032\u02BC\u00B4]", "'", text)
        text = re.sub(r"[\u201C\u201D\u201E\u2033]", '"', text)
        # Replace multiple whitespaces with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _clean_item_title(self, text: str) -> str:
        """
        Clean item title by removing page numbers
        
        Args:
            text: Title text that may contain page numbers
            
        Returns:
            Cleaned title without page numbers
        """
        # Remove trailing page numbers (common patterns)
        # Pattern: digit(s) at the end, possibly after dots or spaces
        text = re.sub(r'\s+\d+\s*$', '', text)  # "Item 1. Business 1" -> "Item 1. Business"
        text = re.sub(r'\.\s*\d+\s*$', '.', text)  # "Business.1" -> "Business."
        text = re.sub(r'\d+\s*$', '', text)  # Remove trailing digits
        return text.strip()
    
    def _extract_item_number(self, text: str) -> Optional[str]:
        """
        Extract item number from text
        
        Args:
            text: Text potentially containing item number
            
        Returns:
            Item number (e.g., "1", "1A", "7") or None
        """        # Common patterns for items
        patterns = [
            r'item\s+(\d{1,2}[A-Za-z]?)\b',  # "Item 1A", "Item 7"
            r'part\s+[IV]+\s*[–-]\s*item\s+(\d{1,2}[A-Za-z]?)\b',  # "Part II - Item 1A"
            r'^\s*(\d{1,2}[A-Za-z]?)(?=[A-Za-z])',  # "1Business", "1ARisk Factors"
            r'^\s*(\d{1,2}[A-Za-z]?)\s*[.:-]\s*[A-Za-z]',  # "1A. Risk Factors", "1: Business"
            r'^\s*(\d{1,2}[A-Za-z]?)\s+[A-Za-z]',  # "1A Risk Factors" (TOC row variant)
        ]
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        return None

    def _extract_item_numbers(self, text: str) -> list[str]:
        """
        Extract all item numbers that appear in a TOC row text.
        Handles compact rows like:
        - "PART II. 5. Market for ..."
        - "1A. Risk Factors"
        - "Item 7A ..."
        - "Items 1 and 2. Business and Properties"
        """
        found: list[str] = []
        seen = set()
        text_lower = text.lower()

        # Handle explicit combined plural rows first (non-standard but seen in filings),
        # e.g. "Items 1 and 2. Business and Properties".
        combo = re.search(
            r'\bitems?\s+(\d{1,2}[a-z]?)\s*(?:[.:])?\s+and\s+(\d{1,2}[a-z]?)\b',
            text_lower,
            re.IGNORECASE,
        )
        if combo:
            for g in (combo.group(1), combo.group(2)):
                token = g.upper()
                if token not in seen:
                    seen.add(token)
                    found.append(token)

        patterns = [
            r'item\s+(\d{1,2}[a-z]?)\b',
            r'part\s+[ivx]+\s*[.:-]?\s*(\d{1,2}[a-z]?)\s*[.:]',
            r'(?<!\d)(\d{1,2}[a-z]?)\s*[.:]\s*[a-z]',
        ]
        for pat in patterns:
            for m in re.finditer(pat, text_lower, re.IGNORECASE):
                token = m.group(1).upper()
                if token not in seen:
                    seen.add(token)
                    found.append(token)
        return found
    
    def _find_toc_table(self, soup: BeautifulSoup) -> Optional[Tag]:
        """
        Find the Table of Contents table in the filing

        Args:
            soup: BeautifulSoup object of the filing
            
        Returns:
            Table element containing TOC or None
        """
        # Look for tables that might be TOC
        tables = soup.find_all('table')
        
        # First, find all potential TOC tables
        potential_toc_tables = []
        
        for table in tables:
            table_text = self._clean_text(table.get_text())
            table_text_lower = table_text.lower()
            
            # Check if table contains TOC indicators
            toc_indicators = [
                'table of contents',
                'index to financial statements',
                'item 1.',
                'item 1a',
                'part i',
                'part ii',
                'item 1 ',  # Match "Item 1 " pattern
            ]
            
            # Count how many item references are in the table
            item_count = len(re.findall(r'item\s+\d+[a-z]?', table_text_lower))
            
            # If table has TOC indicators or many items, it's likely the TOC
            has_toc_indicator = any(indicator in table_text_lower for indicator in toc_indicators)
            
            if (has_toc_indicator or item_count >= 2):
                # Additional validation - check if it has links/anchors
                links = table.find_all('a')
                if links or item_count >= 3:  # Must have links OR at least 3 items
                    potential_toc_tables.append((item_count, table))
        
        # Return the table with the most items (most likely to be the real TOC)
        if potential_toc_tables:
            potential_toc_tables.sort(reverse=True, key=lambda x: x[0])
            return potential_toc_tables[0][1]
        
        return None

    def _get_toc_region_html(self, html_content: str) -> Optional[str]:
        """
        Return a beginning-of-filing region around TOC markers.

        The extractor is TOC-driven by design, so we only parse TOC when an
        explicit TOC marker exists near the start of the filing.
        """
        marker_match = self.toc_marker_pattern.search(html_content)
        if not marker_match:
            return None

        if marker_match.start() > self.max_toc_marker_offset:
            return None

        start = max(0, marker_match.start() - self.toc_region_padding_before)
        end = min(len(html_content), marker_match.start() + self.toc_region_length)
        return html_content[start:end]
    
    def _parse_toc_from_table(self, table: Tag, filing_type: str) -> Dict[str, Dict[str, str]]:
        """
        Parse Table of Contents from a table element
        
        Args:
            table: Table element containing TOC
            filing_type: Type of filing (10-K or 10-Q)
            
        Returns:
            Dictionary mapping item numbers to their anchor links/IDs
        """
        toc_items = {}
        
        # Process all rows
        rows = table.find_all('tr')
        
        for row in rows:
            row_text = self._clean_text(row.get_text())
            item_numbers = self._extract_item_numbers(row_text)
            if not item_numbers:
                item_number = self._extract_item_number(row_text)
                if item_number:
                    item_numbers = [item_number]

            if item_numbers:
                links = row.find_all('a', href=True)
                anchor = None
                for link in links:
                    href = link['href']
                    if '#' in href:
                        anchor = href.split('#')[1]
                    elif href.startswith('#'):
                        anchor = href[1:]
                    if anchor:
                        break

                if not anchor and row.get('id'):
                    anchor = row['id']

                title = self._clean_item_title(row_text)
                if len(title) > 240:
                    title = ""

                for item_number in item_numbers:
                    existing = toc_items.get(item_number)
                    # Never overwrite anchored entry with unanchored entry.
                    if existing and existing.get('anchor') and not anchor:
                        continue
                    if existing and existing.get('anchor') and anchor:
                        continue
                    toc_items[item_number] = {
                        'anchor': anchor,
                        'title': title if title else f'Item {item_number}'
                    }
        
        return toc_items
    
    def _find_toc_from_structure(self, soup: BeautifulSoup, filing_type: str) -> Dict[str, Dict[str, str]]:
        """
        Find TOC by analyzing document structure (fallback method)
        
        Args:
            soup: BeautifulSoup object of the filing
            filing_type: Type of filing (10-K or 10-Q)
            
        Returns:
            Dictionary mapping item numbers to their anchor links/IDs
        """
        toc_items = {}
        
        # Limit search to first 5000 characters to avoid scanning entire document
        # Look for headings that indicate items
        max_items_to_find = 20  # Set a reasonable limit
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'div'], limit=200)
        
        for heading in headings:
            if len(toc_items) >= max_items_to_find:
                break
                
            text = self._clean_text(heading.get_text())
            item_number = self._extract_item_number(text)
            
            if item_number:
                # Look for nearby anchor or ID
                anchor = None
                
                # Check current element
                if heading.get('id'):
                    anchor = heading['id']
                elif heading.get('name'):
                    anchor = heading['name']
                
                # Check for anchor tags nearby
                if not anchor:
                    a_tag = heading.find('a')
                    if a_tag:
                        if a_tag.get('name'):
                            anchor = a_tag['name']
                        elif a_tag.get('id'):
                            anchor = a_tag['id']
                
                # Look at previous siblings for anchors
                if not anchor:
                    prev = heading.find_previous('a', attrs={'name': True})
                    if prev and heading.sourceline and prev.sourceline:
                        if abs(heading.sourceline - prev.sourceline) < 10:
                            anchor = prev['name']
                
                if item_number not in toc_items or not toc_items[item_number].get('anchor'):
                    toc_items[item_number] = {
                        'anchor': anchor,
                        'title': self._clean_item_title(text)
                    }
        
        return toc_items

    def _parse_toc_from_links(self, soup: BeautifulSoup) -> Dict[str, Dict[str, str]]:
        """
        Parse TOC from anchor links (common in inline-XBRL filings where TOC is
        represented as linked item labels instead of a clean table).
        """
        toc_items: Dict[str, Dict[str, str]] = {}
        links = soup.find_all("a", href=True)
        for link in links:
            link_text = self._clean_text(link.get_text(" ", strip=True))

            href = link.get("href", "").strip()
            if not href:
                continue

            anchor = None
            if href.startswith("#"):
                anchor = href[1:]
            elif "#" in href:
                anchor = href.split("#", 1)[1]
            if not anchor:
                continue

            # Prefer richer row/cell context over small inline wrappers.
            # Example: "ITEM 6. [Reserved] 42" often has anchors only on
            # "[Reserved]" and page number, while the item token is in sibling cell text.
            context_text = ""
            item_numbers: list[str] = []
            chosen_context = None
            for tag_name in ["tr", "td", "li", "p", "div"]:
                candidate = link.find_parent(tag_name)
                if not candidate:
                    continue
                candidate_text = self._clean_text(candidate.get_text(" ", strip=True))
                if not candidate_text:
                    continue
                nums = self._extract_item_numbers(candidate_text)
                if not nums:
                    one_ctx = self._extract_item_number(candidate_text)
                    nums = [one_ctx] if one_ctx else []
                if nums:
                    context_text = candidate_text
                    item_numbers = nums
                    chosen_context = candidate
                    break

            if not context_text:
                # Fallback to nearest acceptable container text, then link text.
                container = link.find_parent(["tr", "td", "li", "p", "div"])
                context_text = self._clean_text(container.get_text(" ", strip=True)) if container else link_text

            title_text = context_text if 0 < len(context_text) <= 250 else link_text
            if not item_numbers:
                item_numbers = self._extract_item_numbers(context_text) if context_text else []
            if not item_numbers:
                one = self._extract_item_number(link_text) if link_text else None
                item_numbers = [one] if one else []
            if not item_numbers:
                continue

            for item_number in item_numbers:
                existing = toc_items.get(item_number)
                if existing and existing.get("anchor"):
                    continue
                toc_items[item_number] = {
                    "anchor": anchor,
                    "title": self._clean_item_title(title_text),
                }
        return toc_items
    
    def parse_toc(self, html_content: str, filing_type: str) -> Optional[Dict[str, Dict[str, str]]]:
        """
        Parse Table of Contents from SEC filing HTML
        
        Args:
            html_content: HTML content of the filing
            filing_type: Type of filing (10-K or 10-Q)
            
        Returns:
            Dictionary mapping item numbers to their info, or None if no TOC found
            Format: {
                "1": {"anchor": "item1", "title": "Item 1. Business"},
                "1A": {"anchor": "item1a", "title": "Item 1A. Risk Factors"},
                ...
            }
        """
        toc_region_html = self._get_toc_region_html(html_content)
        has_explicit_marker = toc_region_html is not None
        if not toc_region_html:
            # Some filings have a valid TOC table but no literal TOC marker text.
            # In that case, only attempt table-based detection in the beginning region.
            toc_region_html = html_content[:self.toc_fallback_prefix_length]

        soup = BeautifulSoup(toc_region_html, 'html.parser')
        
        def _merge_missing(base: Dict[str, Dict[str, str]], extra: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
            for k, v in extra.items():
                if k not in base:
                    base[k] = v
                elif (not base[k].get("anchor")) and v.get("anchor"):
                    base[k]["anchor"] = v.get("anchor")
                    if v.get("title"):
                        base[k]["title"] = v.get("title")
            return base

        # Try to find TOC table first
        toc_table = self._find_toc_table(soup)
        
        if toc_table:
            toc_items = self._parse_toc_from_table(toc_table, filing_type)
            if toc_items and len(toc_items) >= 2:
                # Enrich table-derived TOC with link-derived anchors/titles.
                linked_items = self._parse_toc_from_links(soup)
                for k, v in linked_items.items():
                    if k not in toc_items:
                        toc_items[k] = v
                    elif not toc_items[k].get("anchor") and v.get("anchor"):
                        toc_items[k]["anchor"] = v.get("anchor")
                        if v.get("title"):
                            toc_items[k]["title"] = v.get("title")

                anchored_count = sum(1 for v in toc_items.values() if v.get("anchor"))
                if anchored_count >= 2:
                    # Enrich once with a broader prefix scan to recover edge rows
                    # not present in the immediate TOC marker region.
                    broad_soup = BeautifulSoup(html_content[: self.toc_fallback_prefix_length], "html.parser")
                    broad_items = self._parse_toc_from_links(broad_soup)
                    toc_items = _merge_missing(toc_items, broad_items)
                    return toc_items

        # Try parsing linked TOC entries from the TOC region.
        toc_items = self._parse_toc_from_links(soup)
        anchored_count = sum(1 for v in toc_items.values() if v.get("anchor"))
        if toc_items and len(toc_items) >= 5 and anchored_count >= 5:
            broad_soup = BeautifulSoup(html_content[: self.toc_fallback_prefix_length], "html.parser")
            broad_items = self._parse_toc_from_links(broad_soup)
            toc_items = _merge_missing(toc_items, broad_items)
            return toc_items
 
        # Some filings place index/TOC links outside the immediate TOC marker
        # region (e.g., repeated page headers with linked ITEM anchors).
        # Fallback to a larger prefix scan, then full-document link scan.
        broad_soup = BeautifulSoup(html_content[: self.toc_fallback_prefix_length], "html.parser")
        toc_items = self._parse_toc_from_links(broad_soup)
        anchored_count = sum(1 for v in toc_items.values() if v.get("anchor"))
        if toc_items and len(toc_items) >= 5 and anchored_count >= 5:
            return toc_items

        full_soup = BeautifulSoup(html_content, "html.parser")
        toc_items = self._parse_toc_from_links(full_soup)
        anchored_count = sum(1 for v in toc_items.values() if v.get("anchor"))
        if toc_items and len(toc_items) >= 5 and anchored_count >= 5:
            return toc_items

        # If no explicit marker exists, do not perform loose structure fallback.
        if not has_explicit_marker:
            return None

        # Fallback: analyze document structure (but limit search)
        toc_items = self._find_toc_from_structure(soup, filing_type)
        
        # Only return if we found at least 2 items
        if len(toc_items) >= 2:
            return toc_items
        
        return None
    
    def get_item_positions(self, html_content: str, 
                          toc_items: Dict[str, Dict[str, str]]) -> Dict[str, Tuple[int, int]]:
        """
        Get the start and end positions of each item in the HTML
        
        Args:
            html_content: HTML content of the filing
            toc_items: TOC items dictionary
            
        Returns:
            Dictionary mapping item numbers to (start_pos, end_pos) tuples
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        positions = {}
        
        # Preserve TOC appearance order. This is important for combined rows
        # like "Items 1 and 2 ..." where both items share one anchor and should
        # have the same section boundary before the next TOC item (e.g., Item 1A).
        sorted_items = list(toc_items.keys())

        def _anchor_start(anchor_val: Optional[str], search_from: int = 0) -> int:
            if not anchor_val:
                return -1
            anchor_pattern = rf'(?:id|name)\s*=\s*[\'\"]{re.escape(anchor_val)}[\'\"]'
            m = re.search(anchor_pattern, html_content[search_from:], re.IGNORECASE)
            if not m:
                return -1
            pos = search_from + m.start()
            tag_open = html_content.rfind('<', 0, pos)
            return tag_open if tag_open != -1 else pos

        def _find_item_heading_start(item_num: str, lo: int, hi: int) -> int:
            """
            Fallback start finder for items that have no TOC anchor.
            Searches ITEM heading tokens in a bounded window and chooses the
            last match (typically the real section heading before next item).
            """
            if hi <= lo:
                return -1
            window = html_content[lo:hi]
            # Allow tags/non-breaking spaces between ITEM and item number.
            item_pat = re.compile(
                rf'ITEM(?:\s|&nbsp;|&#160;|<[^>]+>){{0,20}}{re.escape(item_num)}(?:\b|[.:])',
                re.IGNORECASE,
            )
            matches = list(item_pat.finditer(window))
            if not matches:
                return -1
            m = matches[-1]
            pos = lo + m.start()
            tag_open = html_content.rfind('<', lo, pos)
            return tag_open if tag_open != -1 else pos

        def _find_next_heading_among(items: list[str], lo: int, hi: int) -> int:
            """
            Find earliest heading occurrence among candidate item numbers in [lo, hi).
            """
            best = -1
            for candidate_item in items:
                pos = _find_item_heading_start(candidate_item, lo, hi)
                if pos == -1:
                    continue
                if best == -1 or pos < best:
                    best = pos
            return best

        def trim_end_at_part_heading(start_pos: int, end_pos: int) -> int:
            """
            If a PART heading appears between current and next TOC anchor,
            trim the current item so PART headers are excluded.
            """
            if end_pos <= start_pos:
                return end_pos
            segment = html_content[start_pos:end_pos]
            match = self.part_heading_html_pattern.search(segment)
            if not match:
                return end_pos

            candidate = start_pos + match.start()
            # Guardrails to avoid truncating on incidental in-text mentions.
            if (candidate - start_pos) < 200:
                return end_pos
            if (end_pos - candidate) > 12000:
                return end_pos

            tag_open = html_content.rfind('<', start_pos, candidate)
            return tag_open if tag_open != -1 else candidate
        
        for i, item_num in enumerate(sorted_items):
            anchor = toc_items[item_num].get('anchor')
            
            # Find the element with this anchor
            start_element = None
            
            if anchor:
                # Look for anchor by name or id
                start_element = soup.find(attrs={'name': anchor})
                if not start_element:
                    start_element = soup.find(attrs={'id': anchor})
            
            # Determine start position in raw HTML (avoid BeautifulSoup re-serialization)
            # Find the anchor id/name attribute, then locate the opening tag
            start_pos = -1
            if anchor:
                start_pos = _anchor_start(anchor, 0)
            else:
                # Fallback when TOC captured item number/title but no anchor.
                prev_anchor_pos = -1
                for j in range(i - 1, -1, -1):
                    prev_anchor = toc_items[sorted_items[j]].get('anchor')
                    prev_anchor_pos = _anchor_start(prev_anchor, 0)
                    if prev_anchor_pos != -1:
                        break
                next_anchor_pos = len(html_content)
                for j in range(i + 1, len(sorted_items)):
                    next_anchor = toc_items[sorted_items[j]].get('anchor')
                    npos = _anchor_start(next_anchor, 0)
                    if npos != -1:
                        next_anchor_pos = npos
                        break
                lo = 0 if prev_anchor_pos == -1 else prev_anchor_pos
                hi = next_anchor_pos
                start_pos = _find_item_heading_start(item_num, lo, hi)

            if start_pos != -1:
                # Find end position (next item or end of document)
                end_pos = len(html_content)

                if i < len(sorted_items) - 1:
                    # If adjacent TOC items share the same anchor (combined section),
                    # skip to the next distinct item for boundary detection.
                    next_idx = i + 1
                    while (
                        next_idx < len(sorted_items)
                        and anchor
                        and toc_items[sorted_items[next_idx]].get('anchor') == anchor
                    ):
                        next_idx += 1

                    if next_idx >= len(sorted_items):
                        next_item = None
                        next_anchor = None
                    else:
                        next_item = sorted_items[next_idx]
                        next_anchor = toc_items[next_item].get('anchor')

                    if next_item and next_anchor:
                        next_anchor_pattern = rf'(?:id|name)\s*=\s*[\'\"]{re.escape(next_anchor)}[\'\"]'
                        next_anchor_match = re.search(next_anchor_pattern, html_content[start_pos:], re.IGNORECASE)
                        if next_anchor_match:
                            # Find the tag opening < before this anchor
                            anchor_pos_in_full = start_pos + next_anchor_match.start()
                            tag_open = html_content.rfind('<', 0, anchor_pos_in_full)
                            if tag_open != -1:
                                end_pos = tag_open
                            else:
                                end_pos = anchor_pos_in_full
                    elif next_item:
                        # Next item has no anchor: fallback to next heading search.
                        # Limit search to before the next anchored item after current.
                        hi = len(html_content)
                        for j in range(next_idx, len(sorted_items)):
                            anc = toc_items[sorted_items[j]].get('anchor')
                            anc_pos = _anchor_start(anc, 0)
                            if anc_pos != -1:
                                hi = anc_pos
                                break
                        # Use only the immediate next TOC item for boundary.
                        # Scanning all future items can truncate early on
                        # in-text references (e.g., "see Item 8") inside current item.
                        heading_pos = _find_item_heading_start(next_item, start_pos + 1, hi)
                        if heading_pos != -1:
                            end_pos = heading_pos
                        elif hi != len(html_content):
                            # If we cannot find an unanchored next-item heading,
                            # fall back to the next anchored item boundary.
                            end_pos = hi
                    else:
                        # No next distinct item found (all remaining share same anchor).
                        # Fall through to end-marker boundary like the last item case.
                        for compiled_pattern in self.end_marker_patterns:
                            marker_match = compiled_pattern.search(html_content[start_pos:])
                            if marker_match:
                                marker_pos_in_full = start_pos + marker_match.start()
                                tag_open = html_content.rfind('<', 0, marker_pos_in_full)
                                if tag_open != -1:
                                    end_pos = tag_open
                                else:
                                    end_pos = marker_pos_in_full
                                break
                    end_pos = trim_end_at_part_heading(start_pos, end_pos)

                else:
                    # For the last item, search for end-marker IDs as boundary
                    # OPTIMIZED: Use pre-compiled patterns instead of compiling on each use
                    for compiled_pattern in self.end_marker_patterns:
                        marker_match = compiled_pattern.search(html_content[start_pos:])
                        if marker_match:
                            # Find the tag opening < before this marker
                            marker_pos_in_full = start_pos + marker_match.start()
                            tag_open = html_content.rfind('<', 0, marker_pos_in_full)
                            if tag_open != -1:
                                end_pos = tag_open
                            else:
                                end_pos = marker_pos_in_full
                            break

                positions[item_num] = (start_pos, end_pos)
        
        return positions


