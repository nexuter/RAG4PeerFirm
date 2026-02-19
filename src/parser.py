"""
SEC Filing Parser - Detects and parses Table of Contents
"""

import re
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup, Tag, NavigableString


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
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing extra whitespace
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
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
        """
        # Common patterns for items
        patterns = [
            r'item\s+(\d+[A-Za-z]?)\b',  # "Item 1A", "Item 7"
            r'part\s+[IV]+\s*[–-]\s*item\s+(\d+[A-Za-z]?)\b',  # "Part II - Item 1A"
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        return None
    
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
            item_number = self._extract_item_number(row_text)
            
            if item_number:
                # Look for links in this row
                links = row.find_all('a', href=True)
                anchor = None
                
                for link in links:
                    href = link['href']
                    # Extract anchor from href
                    if '#' in href:
                        anchor = href.split('#')[1]
                    elif href.startswith('#'):
                        anchor = href[1:]
                    
                    if anchor:
                        break
                
                # If no anchor found, look for nearby anchors or IDs
                if not anchor:
                    # Check if row has an id
                    if row.get('id'):
                        anchor = row['id']
                
                toc_items[item_number] = {
                    'anchor': anchor,
                    'title': self._clean_item_title(row_text)
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
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Try to find TOC table first
        toc_table = self._find_toc_table(soup)
        
        if toc_table:
            toc_items = self._parse_toc_from_table(toc_table, filing_type)
            if toc_items and len(toc_items) >= 2:
                return toc_items
        
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
        
        # Sort items properly (1, 1A, 1B, 2, 3, etc.)
        def item_sort_key(item: str) -> Tuple[int, str]:
            """Sort key for item numbers"""
            match = re.match(r'(\d+)([A-Z]?)', item)
            if match:
                num = int(match.group(1))
                letter = match.group(2) or ''
                return (num, letter)
            return (0, item)
        
        sorted_items = sorted(toc_items.keys(), key=item_sort_key)
        
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
                anchor_pattern = rf'(?:id|name)\s*=\s*[\'\"]{re.escape(anchor)}[\'\"]'
                anchor_match = re.search(anchor_pattern, html_content, re.IGNORECASE)
                if anchor_match:
                    # Go backwards from anchor to find the opening <
                    anchor_start = anchor_match.start()
                    tag_open = html_content.rfind('<', 0, anchor_start)
                    if tag_open != -1:
                        start_pos = tag_open  # Start at the opening < tag itself
                    else:
                        start_pos = anchor_start

            if start_pos != -1:
                # Find end position (next item or end of document)
                end_pos = len(html_content)

                if i < len(sorted_items) - 1:
                    next_item = sorted_items[i + 1]
                    next_anchor = toc_items[next_item].get('anchor')

                    if next_anchor:
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
