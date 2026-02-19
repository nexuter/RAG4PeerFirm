"""
Structure Extractor - Extract hierarchical heading-body pairs from SEC filing items
"""

from bs4 import BeautifulSoup, NavigableString, Tag
from typing import List, Dict, Any, Optional
import re


class StructureExtractor:
    """Extracts hierarchical heading-body structure from SEC filing item HTML"""
    
    def __init__(self):
        """Initialize StructureExtractor"""
        # Heading tags in priority order
        self.heading_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        # Minimum text length to consider as heading
        self.min_heading_length = 3
        self.max_heading_length = 200
        
        # Heading styles (level 1 = bold, level 2 = italic, level 3+ = other styled text)
        self.heading_style_levels = {
            'bold': 1,          # font-weight:700
            'italic': 2,        # font-style:italic
            'underline': 3      # text-decoration:underline
        }
    
    def extract_structure(self, item_html: str) -> List[Dict[str, Any]]:
        """
        Extract hierarchical heading-body structure from item HTML with nesting support
        
        Args:
            item_html: HTML content of the item
            
        Returns:
            List of structured elements with heading, body, and layer information
        """
        soup = BeautifulSoup(item_html, 'lxml')
        
        # Remove script and style tags
        for tag in soup(['script', 'style']):
            tag.decompose()
        
        # Build flat list of all potential elements first
        elements = self._collect_elements(soup)
        
        # Build hierarchical structure from flat list
        structure = self._build_hierarchy(elements)
        
        # If no structure found, return simple text
        if not structure:
            text_content = self._clean_text(soup.get_text())
            if text_content:
                structure.append({
                    'type': 'simple_text',
                    'layer': 1,
                    'heading': None,
                    'body': text_content,
                    'children': []
                })
        
        return structure
    
    def _collect_elements(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Collect all heading and content elements from the soup
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            List of element dictionaries with type, layer, heading, and raw_element
        """
        elements = []
        all_divs = soup.find_all('div')
        
        for div in all_divs:
            # Check if this is a styled heading
            heading_info = self._get_heading_info(div)
            
            if heading_info:
                elements.append({
                    'type': 'heading',
                    'layer': heading_info['level'],
                    'style_type': heading_info['style_type'],
                    'heading': heading_info['text'],
                    'element': div,
                    'is_heading': True
                })
            elif self._is_body_content(div):
                # This is body content between headings
                text = self._clean_text(div.get_text())
                if text and not self._is_page_marker(text):
                    elements.append({
                        'type': 'body',
                        'content': text,
                        'element': div,
                        'is_heading': False
                    })
        
        return elements
    
    def _get_heading_info(self, div: Tag) -> Optional[Dict[str, Any]]:
        """
        Check if div is a heading and return heading information
        
        Args:
            div: Div element to check
            
        Returns:
            Dictionary with heading info or None
        """
        # Check for bold span (level 1)
        bold_span = div.find('span', style=lambda s: s and 'font-weight:700' in s, recursive=False)
        if bold_span:
            text = self._clean_text(div.get_text())
            if text and self.min_heading_length <= len(text) <= self.max_heading_length:
                return {
                    'text': text,
                    'level': 1,
                    'style_type': 'bold'
                }
        
        # Check for italic text (level 2)
        italic_span = div.find('span', style=lambda s: s and 'font-style:italic' in s, recursive=False)
        if italic_span and not bold_span:  # Only if not also bold
            text = self._clean_text(div.get_text())
            if text and self.min_heading_length <= len(text) <= self.max_heading_length:
                # Make sure it's short enough to be a subheading
                if len(text) <= 100:
                    return {
                        'text': text,
                        'level': 2,
                        'style_type': 'italic'
                    }
        
        return None
    
    def _is_body_content(self, div: Tag) -> bool:
        """
        Check if div is body content (not a heading)
        
        Args:
            div: Div element to check
            
        Returns:
            True if this is body content
        """
        # Must not be a heading
        if self._get_heading_info(div):
            return False
        
        # Must have text
        text = div.get_text().strip()
        if not text:
            return False
        
        # Should not be a page marker
        if self._is_page_marker(text):
            return False
        
        return True
    
    def _build_hierarchy(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build hierarchical structure from flat list of elements
        
        Args:
            elements: Flat list of elements
            
        Returns:
            Hierarchical structure with nesting
        """
        if not elements:
            return []
        
        structure = []
        heading_stack = []  # Stack of (level, element) tuples
        
        for elem in elements:
            if elem['is_heading']:
                level = elem['layer']
                
                # Pop headings at same or higher level from stack
                while heading_stack and heading_stack[-1]['layer'] >= level:
                    heading_stack.pop()
                
                # Create heading entry
                heading_entry = {
                    'type': elem['type'],
                    'layer': level,
                    'heading': elem['heading'],
                    'body': '',
                    'children': []
                }
                
                # Add to parent or to root
                if heading_stack:
                    heading_stack[-1]['children'].append(heading_entry)
                else:
                    structure.append(heading_entry)
                
                # Push to stack
                heading_stack.append(heading_entry)
            
            elif elem['is_heading'] == False and heading_stack:
                # This is body content, add to current heading
                if heading_stack[-1]['body']:
                    heading_stack[-1]['body'] += ' ' + elem['content']
                else:
                    heading_stack[-1]['body'] = elem['content']
        
        return structure
    
    def _get_heading_layer(self, tag_name: str, heading_stack: List[tuple]) -> int:
        """
        Determine layer based on heading tag level
        
        Args:
            tag_name: HTML tag name (h1, h2, etc.)
            heading_stack: Current heading hierarchy
            
        Returns:
            Layer number (1-indexed)
        """
        level = int(tag_name[1])  # h1 -> 1, h2 -> 2, etc.
        return level
    
    def _is_page_marker(self, text: str) -> bool:
        """Check if text is a page marker like 'PAGE_BREAK_MARKER' or page numbers"""
        if 'PAGE_BREAK_MARKER' in text:
            return True
        # Check for patterns like "Apple Inc. | 2022 Form 10-K | 1"
        if re.search(r'\|\s*\d{4}\s*Form\s*10-[KQ]\s*\|', text):
            return True
        return False
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ''
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove special characters that are artifacts
        text = re.sub(r'[\xa0\u200b\u200c\u200d\ufeff]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

