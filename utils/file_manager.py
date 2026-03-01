"""
File Manager for handling file operations and directory structure
"""

import json
import os
from typing import Any, Dict


class FileManager:
    """Manages file and directory operations for SEC filings"""

    def __init__(self, base_dir: str = "sec_filings"):
        """
        Initialize FileManager

        Args:
            base_dir: Base directory for storing SEC filings
        """
        self.base_dir = base_dir

    def get_filing_path(self, cik_ticker: str, year: str, filing_type: str, extension: str = "html") -> str:
        """
        Get the file path for a SEC filing

        Args:
            cik_ticker: CIK number or ticker symbol
            year: Filing year
            filing_type: Type of filing (10-K or 10-Q)
            extension: File extension (html or htm)

        Returns:
            Full path to the filing file
        """
        filename = f"{cik_ticker}_{year}_{filing_type}.{extension}"
        path = os.path.join(self.base_dir, cik_ticker, year, filing_type, filename)
        return path

    def get_item_path(self, cik_ticker: str, year: str, filing_type: str, item_number: str) -> str:
        """
        Get the file path for an extracted item

        Args:
            cik_ticker: CIK number or ticker symbol
            year: Filing year
            filing_type: Type of filing (10-K or 10-Q)
            item_number: Item number (e.g., "1", "1A", "7")

        Returns:
            Full path to the item JSON file
        """
        filename = f"{cik_ticker}_{year}_{filing_type}_item{item_number}.json"
        path = os.path.join(self.base_dir, cik_ticker, year, filing_type, "items", filename)
        return path

    def create_directory_structure(self, cik_ticker: str, year: str, filing_type: str) -> None:
        """
        Create directory structure for a filing

        Args:
            cik_ticker: CIK number or ticker symbol
            year: Filing year
            filing_type: Type of filing (10-K or 10-Q)
        """
        filing_dir = os.path.join(self.base_dir, cik_ticker, year, filing_type)
        items_dir = os.path.join(filing_dir, "items")

        os.makedirs(filing_dir, exist_ok=True)
        os.makedirs(items_dir, exist_ok=True)

    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists (explicitly checks for files, not directories)

        Args:
            file_path: Path to the file

        Returns:
            True if file exists, False otherwise
        """
        return os.path.isfile(file_path)

    def save_html(self, file_path: str, content: str) -> None:
        """
        Save HTML content to a file

        Args:
            file_path: Path to save the file
            content: HTML content to save
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    def load_html(self, file_path: str) -> str:
        """
        Load HTML content from a file

        Args:
            file_path: Path to the file

        Returns:
            HTML content
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def save_item_json(self, file_path: str, item_data: Dict[str, Any]) -> None:
        """
        Save extracted item data to JSON

        Args:
            file_path: Path to save the JSON file
            item_data: Dictionary containing item data
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(item_data, f, indent=2, ensure_ascii=False)

    def load_item_json(self, file_path: str) -> Dict[str, Any]:
        """
        Load item data from JSON file

        Args:
            file_path: Path to the JSON file

        Returns:
            Dictionary containing item data
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_json(self, file_path: str) -> Dict[str, Any]:
        """
        Load JSON data from file

        Args:
            file_path: Path to the JSON file

        Returns:
            Dictionary containing JSON data
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_json(self, file_path: str, data: Dict[str, Any]) -> None:
        """
        Save data to JSON file

        Args:
            file_path: Path to save the JSON file
            data: Dictionary containing data to save
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
