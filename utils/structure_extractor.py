"""
Structure Extractor - Extract hierarchical heading-body pairs from SEC filing items
"""

from bs4 import BeautifulSoup, Tag
from typing import List, Dict, Any, Optional
import re
import unicodedata


class StructureExtractor:
    """Extracts hierarchical heading-body structure from SEC filing item HTML"""
    
    def __init__(self):
        """Initialize StructureExtractor"""
        # Heading tags in priority order
        self.heading_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        self.block_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'div', 'table']
        # Minimum text length to consider as heading
        self.min_heading_length = 3
        self.max_heading_length = 220
        self.max_bold_sentence_heading_length = 520
    
    def extract_structure(self, item_html: str, root_heading: Optional[str] = None) -> List[Dict[str, Any]]:
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

        # Apply canonical item-title root heading if provided.
        if root_heading:
            structure = self._apply_root_heading(structure, root_heading)

        return structure
    
    def _collect_elements(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Collect all heading and content elements from the soup
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            List of element dictionaries with type, layer, heading, and raw_element
        """
        elements: List[Dict[str, Any]] = []

        for block in self._iter_blocks_in_order(soup):
            text = self._clean_text(block.get_text())
            if not text:
                continue
            if self._is_page_marker(text):
                continue
            if self._looks_like_noise_line(text):
                continue

            split = self._split_bold_lead(block, text)
            if split is not None:
                heading_text, body_text = split
                heading_info = self._get_heading_info(block, heading_text)
                if heading_info is None:
                    heading_info = {'text': heading_text, 'level': 2, 'style_type': 'bold'}
                elements.append({
                    'type': 'heading',
                    'layer': heading_info['level'],
                    'style_type': heading_info['style_type'],
                    'heading': heading_info['text'],
                    'element': block,
                    'is_heading': True
                })
                if body_text:
                    elements.append({
                        'type': 'body',
                        'content': body_text,
                        'element': block,
                        'is_heading': False
                    })
                continue

            heading_info = self._get_heading_info(block, text)
            if heading_info is not None:
                elements.append({
                    'type': 'heading',
                    'layer': heading_info['level'],
                    'style_type': heading_info['style_type'],
                    'heading': heading_info['text'],
                    'element': block,
                    'is_heading': True
                })
            elif self._is_body_content(block, text):
                elements.append({
                    'type': 'body',
                    'content': text,
                    'element': block,
                    'is_heading': False
                })
        
        return elements

    def _split_bold_lead(self, block: Tag, text: str) -> Optional[tuple]:
        """
        If a block starts with a bold lead-in (e.g., 'Talent Development.')
        followed by regular text, split into heading + body.
        """
        if not text:
            return None
        # Find the first bold-ish descendant.
        lead = None
        for node in block.find_all(True):
            if node.name in {'b', 'strong'}:
                lead = node.get_text()
                break
            style = (node.get('style') or '').lower()
            if re.search(r'font-weight\s*:\s*(bold|[6-9]00)', style):
                lead = node.get_text()
                break
        if not lead:
            return None
        lead_clean = self._clean_text(lead)
        if not lead_clean:
            return None
        full = self._clean_text(text)
        if not full.lower().startswith(lead_clean.lower()):
            return None
        # Require lead to be a short phrase ending with a period
        # OR followed immediately by a separate punctuation span.
        remainder = full[len(lead_clean):].strip()
        if not lead_clean.endswith('.') and not remainder.startswith(('.', ':')):
            return None
        if len(lead_clean) > 120:
            return None
        remainder = remainder.lstrip('.:').strip()
        if not remainder:
            return None
        # Trim trailing punctuation for heading label.
        heading = lead_clean.rstrip('.:').strip()
        return heading, remainder

    def _iter_blocks_in_order(self, soup: BeautifulSoup):
        """
        Yield candidate text blocks in document order.
        Skip container divs that only wrap smaller block elements to avoid duplicates.
        """
        for tag in soup.find_all(self.block_tags):
            if tag.name == 'div':
                if tag.find(['p', 'li', 'td', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    continue
            if tag.name == 'table':
                # Treat table as one plain body block, skip nested cells elsewhere.
                yield tag
                continue
            if tag.find_parent('table'):
                continue
            yield tag

    def _style_blob(self, tag: Tag) -> str:
        """
        Aggregate style attributes from tag and descendants.
        """
        chunks = []
        if tag.get('style'):
            chunks.append(tag.get('style'))
        for child in tag.find_all(True):
            st = child.get('style')
            if st:
                chunks.append(st)
        return " ".join(chunks).lower()

    def _is_item_heading_text(self, text: str) -> bool:
        return bool(re.match(r'^\s*items?\s+\d+[a-z]?\b', text, flags=re.IGNORECASE))

    def _looks_like_noise_line(self, text: str) -> bool:
        t = text.strip().lower()
        if t in {"table of contents", "index to exhibits", "index to financial statements"}:
            return True
        if re.fullmatch(r'part\s+[ivxlcdm]+', t):
            return True
        if re.match(r'^table\s+\d+(\.\d+)*[:.]?\b', t):
            return True
        if re.fullmatch(r'\d{1,4}', t):
            return True
        if re.fullmatch(r'page\s+\d{1,4}(?:\s+of\s+\d{1,4})?', t):
            return True
        return False

    def _get_heading_info(self, block: Tag, text: str) -> Optional[Dict[str, Any]]:
        """
        Determine whether a block is a heading and assign layer.
        
        Args:
            block: Block element
            text: Cleaned block text
            
        Returns:
            Dictionary with heading info or None
        """
        if len(text) < self.min_heading_length:
            return None
        if block.name == 'table':
            return None

        style_blob = self._style_blob(block)
        has_bold = bool(re.search(r'font-weight\s*:\s*(bold|[6-9]00)', style_blob)) or block.find(['b', 'strong']) is not None
        has_italic = bool(re.search(r'font-style\s*:\s*italic', style_blob)) or block.find(['i', 'em']) is not None
        has_underline = bool(re.search(r'text-decoration\s*:\s*underline', style_blob))
        is_center = bool(re.search(r'text-align\s*:\s*center', style_blob)) or (str(block.get('align', '')).lower() == 'center')

        if self._is_item_heading_text(text):
            return {'text': text, 'level': 1, 'style_type': 'item'}

        # Avoid false layer-3 headings where only a person's name is bolded in
        # an executive-officer biography sentence (e.g., "Mr. X is ...").
        if has_bold and self._is_name_intro_sentence(text):
            return None
        # Avoid treating bullet list items as headings when only the bullet is bold.
        if has_bold and self._bold_only_bullet(block):
            return None

        # Length guardrails:
        # - keep normal headings bounded to avoid classifying paragraphs as headings
        # - allow longer bold sentence headings (common in Item 1A risk factors)
        if len(text) > self.max_heading_length:
            if not (has_bold and text.endswith('.') and len(text) <= self.max_bold_sentence_heading_length):
                return None

        # Heuristic heading score (style + shape)
        score = 0
        if has_bold:
            score += 2
        if has_italic or has_underline:
            score += 1
        if is_center:
            score += 1
        if len(text.split()) <= 18:
            score += 1

        # Title-like forms: ALL CAPS or short title without trailing punctuation.
        letters = re.sub(r'[^A-Za-z]', '', text)
        upper_ratio = (sum(ch.isupper() for ch in letters) / len(letters)) if letters else 0.0
        if upper_ratio >= 0.60:
            score += 1
        if re.match(r'^[A-Z][A-Za-z0-9,&/\-\'(). ]+$', text) and not text.endswith('.'):
            score += 1

        # Long sentence-like lines are usually body, not heading, unless explicitly bold.
        if len(text) > 140 and text.endswith('.') and not has_bold:
            score -= 2

        if score < 3 and not has_bold:
            return None

        # Bold sentence-style headings (common in risk factor sections)
        # should be captured as deeper layer headings even when long.
        if has_bold and text.endswith('.') and len(text) <= 260:
            level = 3
            style_type = 'bold_sentence'
        elif has_bold and not text.endswith('.') and self._looks_like_titlecase_heading(text):
            level = 2
            style_type = 'bold'
        elif has_bold and (is_center or upper_ratio >= 0.60 or len(text) <= 90):
            level = 2
            style_type = 'bold'
        elif has_italic or has_underline:
            level = 3
            style_type = 'italic' if has_italic else 'underline'
        else:
            level = 3
            style_type = 'styled'

        return {'text': text, 'level': level, 'style_type': style_type}

    def _is_name_intro_sentence(self, text: str) -> bool:
        """
        Detect biography-like body sentences that often start with a person name
        and are partially bolded, but are not structural headings.
        """
        t = self._clean_text(text)
        if not t or len(t) < 40 or len(t) > 520:
            return False
        # Keep true label-like headings, e.g., "Name: ...".
        if ":" in t[:80]:
            return False

        patterns = [
            r'^(Mr|Ms|Mrs|Dr)\.\s+[A-Z][A-Za-z\'\-]+(?:\s+[A-Z][A-Za-z\'\-]+){0,3}\s+is\b',
            r'^[A-Z][A-Za-z\'\-]+(?:\s+[A-Z][A-Za-z\'\-]+){1,4}\s*,\s*\d{1,3}\s*,?\s+has\b',
            r'^[A-Z][A-Za-z\'\-]+(?:\s+[A-Z][A-Za-z\'\-]+){1,4}\s+is\b',
        ]
        return any(re.match(p, t) for p in patterns)

    def _looks_like_titlecase_heading(self, text: str) -> bool:
        """
        Detect title-case headings that are bold and long but not sentences.
        """
        t = self._clean_text(text)
        if not t or t.endswith('.'):
            return False
        if len(t) > 260:
            return False
        words = re.findall(r"[A-Za-z][A-Za-z'\\-]*", t)
        if len(words) < 4:
            return False
        capped = sum(1 for w in words if w[0].isupper())
        return (capped / max(1, len(words))) >= 0.6

    def _bold_only_bullet(self, block: Tag) -> bool:
        """
        Detect cases where a bullet is bold but the actual sentence is not.
        """
        found_bold = False
        for node in block.find_all(True):
            style = (node.get('style') or '').lower()
            is_bold = node.name in {'b', 'strong'} or bool(re.search(r'font-weight\s*:\s*(bold|[6-9]00)', style))
            if not is_bold:
                continue
            found_bold = True
            txt = self._clean_text(node.get_text())
            if txt and txt not in {'•'}:
                return False
        return found_bold

    def _is_body_content(self, block: Tag, text: str) -> bool:
        """
        Check if block is body content.
        """
        if self._get_heading_info(block, text):
            return False
        if not text:
            return False
        if self._is_page_marker(text):
            return False
        if self._looks_like_noise_line(text):
            return False
        return True

    def _extract_item_token(self, title: str) -> Optional[str]:
        m = re.search(r'items?\s+(\d+[a-z]?)', title or '', flags=re.IGNORECASE)
        return m.group(1).upper() if m else None

    def _is_item_heading_node(self, node: Dict[str, Any], token: Optional[str]) -> bool:
        if node.get('type') != 'heading':
            return False
        h = str(node.get('heading') or '')
        if not token:
            return bool(re.match(r'^\s*items?\s+\d+[a-z]?\b', h, flags=re.IGNORECASE))
        return bool(re.match(rf'^\s*items?\s+{re.escape(token)}\b', h, flags=re.IGNORECASE))

    def _bump_layers(self, nodes: List[Dict[str, Any]], min_layer: int = 2) -> None:
        for n in nodes:
            if n.get('type') == 'heading':
                n['layer'] = max(int(n.get('layer', min_layer)), min_layer)
            self._bump_layers(n.get('children') or [], min_layer=min_layer + 1)

    def _apply_root_heading(self, structure: List[Dict[str, Any]], root_heading: str) -> List[Dict[str, Any]]:
        root_title = self._clean_text(root_heading)
        if not root_title:
            return structure

        token = self._extract_item_token(root_title)
        root_body_parts: List[str] = []
        root_children: List[Dict[str, Any]] = []

        for node in structure:
            if self._is_item_heading_node(node, token):
                body = (node.get('body') or '').strip()
                if body:
                    root_body_parts.append(self._strip_redundant_root_prefix(body, root_title))
                root_children.extend(node.get('children') or [])
            else:
                root_children.append(node)

        # Deduplicate repeated top-level headings.
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for node in root_children:
            key = (node.get('type'), int(node.get('layer', 0)), self._clean_text(str(node.get('heading') or '')).lower())
            if node.get('type') == 'heading' and key in seen and not (node.get('body') or '').strip() and not (node.get('children') or []):
                continue
            seen.add(key)
            deduped.append(node)
        root_children = deduped

        # Fold top-level simple_text into root body and remove it from children.
        kept_children: List[Dict[str, Any]] = []
        for node in root_children:
            if node.get('type') == 'simple_text':
                txt = self._clean_text(str(node.get('body') or ''))
                txt = self._strip_redundant_root_prefix(txt, root_title)
                if txt:
                    root_body_parts.append(txt)
                continue
            kept_children.append(node)
        root_children = kept_children

        self._bump_layers(root_children, min_layer=2)

        root_body = ' '.join(part for part in root_body_parts if part).strip() or None
        if root_body:
            root_body = self._strip_redundant_root_prefix(root_body, root_title) or None

        return [{
            'type': 'heading',
            'layer': 1,
            'heading': root_title,
            'body': root_body,
            'children': root_children
        }]

    def _strip_redundant_root_prefix(self, text: str, root_title: str) -> str:
        """
        Remove repeated item title prefix from body text when simple_text starts
        with the same heading as root.
        """
        txt = self._clean_text(text)
        if not txt:
            return txt
        def norm(s: str) -> str:
            s = re.sub(r"['’]", '', s)
            s = re.sub(r'[^A-Za-z0-9]+', ' ', s)
            return re.sub(r'\s+', ' ', s).strip().lower()

        def strip_prefix(original: str, prefix_norm: str) -> Optional[str]:
            if not prefix_norm:
                return None
            norm_len = 0
            last_index = None
            prev_space = False
            for i, ch in enumerate(original):
                if ch.isalnum():
                    norm_len += 1
                    prev_space = False
                elif ch in "'’":
                    # ignore apostrophes for normalization
                    pass
                else:
                    if not prev_space and norm_len > 0:
                        norm_len += 1
                        prev_space = True
                if norm_len >= len(prefix_norm):
                    last_index = i + 1
                    break
            if last_index is None:
                return None
            return original[last_index:].lstrip(" .:-|,;/")

        root_clean = root_title or ''
        root_no_item = re.sub(r'^\s*item\s+\d+[a-z]?\s*\.?\s*', '', root_clean, flags=re.IGNORECASE)
        candidates = [root_clean, root_no_item]
        for cand in candidates:
            cand_norm = norm(cand)
            if cand_norm and norm(txt).startswith(cand_norm):
                stripped = strip_prefix(txt, cand_norm)
                if stripped is not None:
                    return stripped.strip()
        return txt

    def _build_hierarchy(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build hierarchical structure from flat list of elements.
        """
        if not elements:
            return []

        structure: List[Dict[str, Any]] = []
        heading_stack: List[Dict[str, Any]] = []

        for elem in elements:
            if elem['is_heading']:
                level = int(elem['layer'])

                while heading_stack and heading_stack[-1]['layer'] >= level:
                    heading_stack.pop()

                heading_entry = {
                    'type': 'heading',
                    'layer': level,
                    'heading': elem['heading'],
                    'body': None,
                    'children': []
                }

                if heading_stack:
                    heading_stack[-1]['children'].append(heading_entry)
                else:
                    structure.append(heading_entry)

                heading_stack.append(heading_entry)
            else:
                if not heading_stack:
                    # Keep pre-heading text if present
                    if structure and structure[-1].get('type') == 'simple_text':
                        structure[-1]['body'] += ' ' + elem['content']
                    else:
                        structure.append({
                            'type': 'simple_text',
                            'layer': 1,
                            'heading': None,
                            'body': elem['content'],
                            'children': []
                        })
                else:
                    cur = heading_stack[-1]
                    if cur['body'] is None:
                        cur['body'] = elem['content']
                    else:
                        cur['body'] += ' ' + elem['content']

        return structure

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
        text = unicodedata.normalize("NFKC", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2013', '-').replace('\u2014', '-')
        text = re.sub(r'[\u2022\u25CF\u25A0\u25AA\u25E6\u2043\u2219]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove special characters that are artifacts
        text = re.sub(r'[\xa0\u200b\u200c\u200d\ufeff]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    # Legacy helper kept for compatibility; unused in current style-based layering.
    def _get_heading_layer(self, tag_name: str, heading_stack: List[tuple]) -> int:
        level = int(tag_name[1])  # h1 -> 1, h2 -> 2, etc.
        return level

    # Backward-compatible signature; unused by current collector.
    def _is_body_content_legacy(self, div: Tag) -> bool:
        text = div.get_text().strip()
        return bool(text) and not self._is_page_marker(text)

