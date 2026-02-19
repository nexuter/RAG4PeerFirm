"""Load item text from extracted 10-K items."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .text_utils import join_parts, normalize_text


def _default_item_path(base_dir: str, cik: str, year: str, filing_type: str, item: str) -> str:
    filename = f"{cik}_{year}_{filing_type}_item{item}.json"
    return os.path.join(base_dir, cik, year, filing_type, "items", filename)


def _default_structure_path(base_dir: str, cik: str, year: str, filing_type: str, item: str) -> str:
    filename = f"{cik}_{year}_{filing_type}_item{item}_xtr.json"
    return os.path.join(base_dir, cik, year, filing_type, "items", filename)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _flatten_structure(structure: List[Dict[str, Any]]) -> str:
    parts: List[str] = []

    def walk(nodes: List[Dict[str, Any]]) -> None:
        for node in nodes:
            heading = node.get("heading")
            body = node.get("body")
            if heading:
                parts.append(str(heading))
            if body:
                parts.append(str(body))
            children = node.get("children") or []
            if children:
                walk(children)

    walk(structure)
    return join_parts(parts)


def load_item_text(
    base_dir: str,
    cik: str,
    year: str,
    filing_type: str,
    item: str,
) -> Tuple[Optional[str], Optional[str]]:
    item_path = _default_item_path(base_dir, cik, year, filing_type, item)
    if os.path.isfile(item_path):
        payload = _load_json(item_path)
        text = payload.get("text_content")
        if text:
            return normalize_text(text), item_path

    structure_path = _default_structure_path(base_dir, cik, year, filing_type, item)
    if os.path.isfile(structure_path):
        payload = _load_json(structure_path)
        structure = payload.get("structure")
        if isinstance(structure, list):
            text = _flatten_structure(structure)
            if text:
                return text, structure_path

    return None, None
