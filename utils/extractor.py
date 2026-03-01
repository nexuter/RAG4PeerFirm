"""
Item Extractor - Extracts individual items from SEC filings
"""

import re
import unicodedata
from typing import Dict, List, Any
from bs4 import BeautifulSoup
from .parser import SECParser


class ItemExtractor:
    """Extracts specific items from SEC filings using TOC information"""
    
    def __init__(self):
        """Initialize ItemExtractor"""
        self.parser = SECParser()
        self._page_break_marker = "PAGE_BREAK_MARKER"
        
        # Pre-compile artifact patterns for faster detection
        self.artifact_patterns = [
            re.compile(r"^\d{1,3}$"),
            re.compile(r"^[a-z\s]+(?:inc|corp|ltd|llc|co)\.?$"),
            re.compile(r"^[a-z]+ (?:inc|corp|ltd)\s*\|\s*\d{4}"),
            re.compile(r"^(?:inc|form|10-?k)\s*\|"),
        ]
        self.artifact_phrases = {
            "table of contents", "page", "form", "10-k", "10-q", "10-a",
            "form 10-k summary", "not applicable"
        }
        self._zero_width_chars = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u2060"]

    def _normalize_unicode(self, text: str) -> str:
        """
        Remove invisible unicode artifacts and normalize common smart punctuation.
        """
        text = unicodedata.normalize("NFKC", text)
        for ch in self._zero_width_chars:
            text = text.replace(ch, "")
        # Remove any remaining unicode formatting/control artifacts.
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
        # Remove common bullet/ornament symbols that pollute extracted prose.
        text = re.sub(r"[\u2022\u25CF\u25A0\u25AA\u25E6\u2043\u2219]", " ", text)
        return (
            text.replace("\u2018", "'")
            .replace("\u2019", "'")
            .replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2013", "-")
            .replace("\u2014", "-")
        )

    def _remove_line_artifacts(self, text: str) -> str:
        """
        Remove generic line-level artifacts such as page numbers and repeated headers.
        """
        raw_lines = [ln.strip() for ln in text.splitlines()]
        lines: List[str] = []

        def prev_non_empty(idx: int) -> str:
            j = idx - 1
            while j >= 0:
                if raw_lines[j]:
                    return raw_lines[j]
                j -= 1
            return ""

        def next_non_empty(idx: int) -> str:
            j = idx + 1
            while j < len(raw_lines):
                if raw_lines[j]:
                    return raw_lines[j]
                j += 1
            return ""

        for idx, line in enumerate(raw_lines):
            if not line:
                continue
            low = line.lower()
            if low == "table of contents":
                continue
            if re.fullmatch(r"page\s+\d{1,4}(?:\s+of\s+\d{1,4})?", low):
                continue
            lines.append(line)
        cleaned = "\n".join(lines)
        # Remove inline TOC headers if they survived line filtering.
        cleaned = re.sub(r"\btable of contents\b", " ", cleaned, flags=re.IGNORECASE)
        return cleaned

    def _postprocess_item_text(self, text: str, item_number: str) -> str:
        """
        Final text cleanup with item-aware rules:
        - Remove leading page-number noise before the item heading
        - Treat early 'Not applicable.' as terminal item content
        """
        out = text.strip()

        # Remove leading page number immediately before item heading
        out = re.sub(r'^\s*\d{1,3}\s+(?=ITEM\s+\d+[A-Z]?\b)', '', out, flags=re.IGNORECASE)

        # Trim any preamble before first explicit item heading for this item
        item_head = re.search(rf'ITEM\s+{re.escape(item_number)}\s*[.:]?\s*', out, flags=re.IGNORECASE)
        if item_head:
            out = out[item_head.start():]

        # Two-rule policy:
        # 1) Cut only when "Not applicable." or "None." appears first after item title.
        # 2) Do not cut when marker appears later in the item text.
        # "appears first" is implemented as:
        # - marker found in early window right after "ITEM X"
        # - no sentence-ending punctuation before marker in that early window
        heading = re.search(
            rf"ITEM\s+{re.escape(item_number)}\s*[.:]?\s*",
            out,
            flags=re.IGNORECASE,
        )
        if heading:
            after = out[heading.end() : heading.end() + 420]
            marker = re.search(r"\b(?:Not\s+applicable|None)\b\.", after, flags=re.IGNORECASE)
            if marker:
                prefix = after[: marker.start()]
                # If there is no earlier sentence-ending punctuation,
                # treat marker as first terminal sentence and cut there.
                if not re.search(r"[.!?;:]", prefix):
                    out = out[: heading.end() + marker.end()].strip()

        # Drop trailing standalone page number tokens.
        out = re.sub(r'\s+\d{1,3}\s*$', '', out)
        return out.strip()

    def _strip_headers_footers(self, text: str) -> str:
        """
        Remove repeating headers/footers and page artifacts from text.
        OPTIMIZED: Simple heuristic, much faster than complex multi-phrase checking.
        """
        # Split by page break markers to handle multi-page items
        pages = text.split(self._page_break_marker)
        pages = [p.strip() for p in pages if p.strip()]
        
        if not pages:
            return text
        
        cleaned_pages = []
        
        for page in pages:
            words = page.split()
            if not words:
                cleaned_pages.append(page)
                continue
            
            # Remove trailing artifacts (keep at least 10 words)
            while len(words) > 10:
                last_word = words[-1].lower()
                
                # Check if last word is an artifact
                if last_word.isdigit() or last_word in self.artifact_phrases or last_word == '|':
                    words.pop()
                # Check if last 2 words match artifact phrase
                elif len(words) >= 2 and ' '.join(words[-2:]).lower() in self.artifact_phrases:
                    words.pop()
                    words.pop()
                # Check if last word matches pre-compiled patterns
                elif any(pattern.match(last_word) for pattern in self.artifact_patterns):
                    words.pop()
                else:
                    break
            
            # Remove leading artifacts (keep at least 10 words)
            while len(words) > 10:
                first_word = words[0].lower()
                
                if first_word.isdigit() or first_word in self.artifact_phrases or first_word == '|':
                    words.pop(0)
                elif len(words) >= 2 and ' '.join(words[:2]).lower() in self.artifact_phrases:
                    words.pop(0)
                    words.pop(0)
                elif any(pattern.match(first_word) for pattern in self.artifact_patterns):
                    words.pop(0)
                else:
                    break
            
            cleaned_pages.append(' '.join(words) if words else '')
        
        result = ' '.join(p for p in cleaned_pages if p.strip())
        return result.strip()
    
    def _html_to_text(self, html_content: str) -> str:
        """
        Convert HTML to plain text
        
        Args:
            html_content: HTML content
            
        Returns:
            Plain text
        """
        html_with_breaks = re.sub(
            r"<hr[^>]*page-break-after\s*:\s*always[^>]*>",
            f"\n{self._page_break_marker}\n",
            html_content,
            flags=re.IGNORECASE,
        )

        soup = BeautifulSoup(html_with_breaks, 'lxml')
        
        # Remove script and style elements
        for script in soup(['script', 'style']):
            script.decompose()
        
        text = soup.get_text(separator='\n')
        text = self._normalize_unicode(text)
        text = self._remove_line_artifacts(text)
        text = ' '.join(text.split())
        text = self._strip_headers_footers(text)
        return text
    
    def _clean_html(self, html_content: str) -> str:
        """
        Clean HTML content while preserving structure
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            Cleaned HTML content
        """
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Remove script and style tags but keep other formatting
        for tag in soup(['script', 'style']):
            tag.decompose()
        
        return str(soup)
    
    def extract_item(self, html_content: str, item_number: str, 
                    toc_items: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
        """
        Extract a specific item from the filing
        OPTIMIZED: Parse once, extract both HTML and text from same parse.
        
        Args:
            html_content: HTML content of the filing
            item_number: Item number to extract (e.g., "1", "1A", "7")
            toc_items: TOC items dictionary from parser
            
        Returns:
            Dictionary containing:
                - item_number: str
                - item_title: str
                - html_content: str (original HTML of the item)
                - text_content: str (plain text of the item)
        """
        if item_number not in toc_items:
            raise ValueError(f"Item {item_number} not found in TOC")
        
        # Get positions of all items
        positions = self.parser.get_item_positions(html_content, toc_items)
        
        if item_number not in positions:
            raise ValueError(f"Could not locate Item {item_number} in the document")
        
        start_pos, end_pos = positions[item_number]
        
        # Extract HTML content for this item
        item_html = html_content[start_pos:end_pos]
        
        # OPTIMIZED: Parse once, extract both HTML and text
        html_with_breaks = re.sub(
            r"<hr[^>]*page-break-after\s*:\s*always[^>]*>",
            f"\n{self._page_break_marker}\n",
            item_html,
            flags=re.IGNORECASE,
        )
        
        soup = BeautifulSoup(html_with_breaks, 'lxml')
        
        # Remove script and style tags
        for tag in soup(['script', 'style']):
            tag.decompose()
        
        # Extract clean HTML
        item_html_clean = str(soup)
        
        # Extract text from the same parsed soup
        item_text = soup.get_text(separator='\n')
        item_text = self._normalize_unicode(item_text)
        item_text = self._remove_line_artifacts(item_text)
        item_text = ' '.join(item_text.split())  # Collapse whitespace
        item_text = self._strip_headers_footers(item_text)
        item_text = self._postprocess_item_text(item_text, item_number)
        
        return {
            'item_number': item_number,
            'item_title': toc_items[item_number].get('title', f'Item {item_number}'),
            'html_content': item_html_clean,
            'text_content': item_text
        }
    
    def extract_items(self, html_content: str, item_numbers: List[str], 
                     toc_items: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
        """
        Extract multiple items from the filing
        
        Args:
            html_content: HTML content of the filing
            item_numbers: List of item numbers to extract
            toc_items: TOC items dictionary from parser
            
        Returns:
            Dictionary mapping item numbers to extracted item data
        """
        extracted_items = {}
        
        for item_number in item_numbers:
            try:
                item_data = self.extract_item(html_content, item_number, toc_items)
                extracted_items[item_number] = item_data
            except Exception as e:
                # Log the error but continue with other items
                extracted_items[item_number] = {
                    'error': str(e)
                }
        
        return extracted_items
    
    def extract_all_items(self, html_content: str, 
                         toc_items: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
        """
        Extract all items found in TOC
        
        Args:
            html_content: HTML content of the filing
            toc_items: TOC items dictionary from parser
            
        Returns:
            Dictionary mapping item numbers to extracted item data
        """
        item_numbers = list(toc_items.keys())
        return self.extract_items(html_content, item_numbers, toc_items)

