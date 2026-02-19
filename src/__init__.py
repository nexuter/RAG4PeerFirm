"""
SEC EDGAR Item Extractor Package
Extracts specific items from SEC EDGAR 10-K and 10-Q filings.
"""

__version__ = "1.0.0"
__author__ = "RAG4PeerFirm Contributors"

from .itemextraction.downloader import SECDownloader
from .itemextraction.parser import SECParser
from .itemextraction.extractor import ItemExtractor

__all__ = ['SECDownloader', 'SECParser', 'ItemExtractor']
