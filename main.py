"""
ItemXtractor - Main script for extracting items from SEC EDGAR filings

This script downloads SEC filings (10-K, 10-Q) and extracts specific items
using the Table of Contents to locate each item within the filing.
"""

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union
from src.downloader import SECDownloader
from src.parser import SECParser
from src.extractor import ItemExtractor
from src.structure_extractor import StructureExtractor
from src.index_parser import SECIndexParser
from utils.logger import ExtractionLogger
from utils.file_manager import FileManager
from config import ITEMS_10K, ITEMS_10Q


class ItemXtractor:
    """Main class for extracting items from SEC EDGAR filings"""
    
    def __init__(self, base_dir: str = "sec_filings", log_dir: str = "logs"):
        """
        Initialize ItemXtractor
        
        Args:
            base_dir: Base directory for storing SEC filings
            log_dir: Directory for log files
        """
        self.downloader = SECDownloader()
        self.parser = SECParser()
        self.extractor = ItemExtractor()
        self.structure_extractor = StructureExtractor()
        self.index_parser = SECIndexParser()
        self.file_manager = FileManager(base_dir)
        self.logger = ExtractionLogger(log_dir)
        self.logger_lock = threading.Lock()  # For thread-safe logging
    
    def _get_available_items(self, filing_type: str) -> List[str]:
        """
        Get list of available items for a filing type
        
        Args:
            filing_type: Type of filing (10-K or 10-Q)
            
        Returns:
            List of available item numbers
        """
        if filing_type == "10-K":
            return list(ITEMS_10K.keys())
        elif filing_type == "10-Q":
            return list(ITEMS_10Q.keys())
        else:
            return []
    def _extract_and_save_item(self, item_number: str, html_content: str, 
                               toc_items: dict, cik_ticker: str, year: str,
                               filing_type: str, filing_record: dict) -> tuple:
        """
        Extract and save a single item (worker method for parallel processing)
        
        Args:
            item_number: Item number to extract
            html_content: HTML content of the filing
            toc_items: TOC items dictionary
            cik_ticker: CIK or ticker symbol
            year: Filing year
            filing_type: Type of filing (10-K or 10-Q)
            filing_record: Filing record for logging
            
        Returns:
            Tuple of (item_number, success, error_message)
        """
        try:
            item_data = self.extractor.extract_item(
                html_content, item_number, toc_items
            )
            
            # Save to JSON
            item_path = self.file_manager.get_item_path(
                cik_ticker, year, filing_type, item_number
            )
            self.file_manager.save_item_json(item_path, item_data)
            
            # Thread-safe logging
            with self.logger_lock:
                self.logger.log_item_extraction(filing_record, item_number, True)
            
            # Extract structure automatically
            try:
                structure = self.structure_extractor.extract_structure(item_data['html_content'])
                
                # Save structure to *_xtr.json
                xtr_filename = f"{cik_ticker}_{year}_{filing_type}_item{item_number}_xtr.json"
                xtr_path = os.path.join(
                    os.path.dirname(item_path),
                    xtr_filename
                )
                
                xtr_data = {
                    'ticker': cik_ticker,
                    'year': year,
                    'filing_type': filing_type,
                    'item_number': item_number,
                    'structure': structure
                }
                
                self.file_manager.save_json(xtr_path, xtr_data)
                
                with self.logger_lock:
                    self.logger.info(
                        f"Extracted structure for Item {item_number}: {len(structure)} elements"
                    )
            except Exception as e:
                with self.logger_lock:
                    self.logger.warning(
                        f"Structure extraction failed for Item {item_number}: {str(e)}"
                    )
            
            return (item_number, True, None)
        except Exception as e:
            # Thread-safe logging
            with self.logger_lock:
                self.logger.log_item_extraction(
                    filing_record, item_number, False, error=str(e)
                )
            return (item_number, False, str(e))
    
    def process_filing(self, cik_ticker: str, year: str, filing_type: str,
                      items: Optional[List[str]] = None) -> bool:
        """
        Process a single SEC filing
        
        Args:
            cik_ticker: CIK number or ticker symbol
            year: Filing year
            filing_type: Type of filing (10-K or 10-Q)
            items: List of item numbers to extract (None = extract all)
            
        Returns:
            True if successful, False otherwise
        """
        # Convert ticker to CIK for consistent folder naming
        try:
            cik = self.downloader.get_cik(cik_ticker)
            original_identifier = cik_ticker
        except Exception as e:
            self.logger.error(f"Failed to resolve {cik_ticker}: {str(e)}")
            return False
        
        # Start logging for this filing (use original identifier for display)
        filing_record = self.logger.log_filing_start(original_identifier, year, filing_type)
        self.logger.info(
            f"Filing worker: {threading.current_thread().name} | {original_identifier} ({cik}) {filing_type} {year}"
        )
        
        try:
            # Create directory structure using CIK
            self.file_manager.create_directory_structure(cik, year, filing_type)
            
            # Determine file path - try both extensions (use CIK for folder)
            filing_path_html = self.file_manager.get_filing_path(cik, year, filing_type, 'html')
            filing_path_htm = self.file_manager.get_filing_path(cik, year, filing_type, 'htm')
            
            # Check if file already exists (explicitly checking for FILES, not directories)
            if self.file_manager.file_exists(filing_path_html):
                self.logger.info(f"File found: {filing_path_html}")
                filing_path = filing_path_html
                html_content = self.file_manager.load_html(filing_path)
                self.logger.log_download(filing_record, True, skipped=True)
            elif self.file_manager.file_exists(filing_path_htm):
                self.logger.info(f"File found: {filing_path_htm}")
                filing_path = filing_path_htm
                html_content = self.file_manager.load_html(filing_path)
                self.logger.log_download(filing_record, True, skipped=True)
            else:
                # Files don't exist, so download them
                self.logger.info(f"Files not found. Will attempt download:")
                self.logger.info(f"  Looking for: {filing_path_html}")
                self.logger.info(f"  Or: {filing_path_htm}")
                
                # Download the filing
                try:
                    html_content, extension, downloaded_cik = self.downloader.download_filing(
                        cik_ticker, filing_type, year
                    )
                    
                    # Use CIK for file path
                    filing_path = self.file_manager.get_filing_path(
                        cik, year, filing_type, extension
                    )
                    self.file_manager.save_html(filing_path, html_content)
                    self.logger.log_download(filing_record, True, skipped=False)
                except Exception as e:
                    self.logger.log_download(filing_record, False, error=str(e))
                    self.logger.log_filing_complete(filing_record)
                    return False
            
            # Parse Table of Contents
            try:
                toc_items = self.parser.parse_toc(html_content, filing_type)
                
                if toc_items:
                    self.logger.log_toc_detection(filing_record, True)
                else:
                    self.logger.log_toc_detection(filing_record, False)
                    self.logger.log_filing_complete(filing_record)
                    return False
                    
            except Exception as e:
                self.logger.log_toc_detection(filing_record, False, error=str(e))
                self.logger.log_filing_complete(filing_record)
                return False
            
            # Determine which items to extract
            if items is None:
                # Extract all items found in TOC
                items_to_extract = list(toc_items.keys())
            else:
                # Only extract requested items that exist in TOC
                items_to_extract = [item for item in items if item in toc_items]
            
            # Extract items sequentially for a single filing (use CIK for folder paths)
            for item_number in items_to_extract:
                self._extract_and_save_item(
                    item_number, html_content, toc_items,
                    cik, year, filing_type, filing_record
                )
            
            self.logger.log_filing_complete(filing_record)
            return True
            
        except Exception as e:
            filing_record['errors'].append(f"Unexpected error: {str(e)}")
            self.logger.error(f"Unexpected error processing filing: {str(e)}")
            self.logger.log_filing_complete(filing_record)
            return False
    
    def extract(self, cik_tickers: Union[str, List[str]], 
                filing_types: Union[str, List[str]],
                years: Union[str, int, List[Union[str, int]]],
                items: Optional[List[str]] = None,
                max_workers: int = 4) -> str:
        """
        Extract items from SEC filings
        
        Args:
            cik_tickers: CIK number(s) or ticker symbol(s)
            filing_types: Filing type(s) (10-K, 10-Q)
            years: Year(s) to extract
            items: List of item numbers to extract (None = extract all)
            max_workers: Number of worker threads for parallel extraction
            
        Returns:
            JSON report string
        """
        # Normalize inputs to lists
        if isinstance(cik_tickers, str):
            cik_tickers = [cik_tickers]
        if isinstance(filing_types, str):
            filing_types = [filing_types]
        if isinstance(years, (str, int)):
            years = [str(years)]
        else:
            years = [str(year) for year in years]
        
        # Log parameters
        self.logger.set_parameters(
            cik_tickers=cik_tickers,
            filing_types=filing_types,
            years=years,
            items=items if items else "all",
            workers=max_workers
        )
        
        # Build filing tasks
        tasks = []
        for cik_ticker in cik_tickers:
            for filing_type in filing_types:
                for year in years:
                    tasks.append((cik_ticker, year, filing_type))
        
        # Parallelize across filings only when multiple filings are requested
        if len(tasks) > 1 and max_workers > 1:
            with ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="filing-worker"
            ) as executor:
                futures = []
                for cik_ticker, year, filing_type in tasks:
                    futures.append(
                        executor.submit(self.process_filing, cik_ticker, year, filing_type, items)
                    )
                for future in futures:
                    future.result()
        else:
            for cik_ticker, year, filing_type in tasks:
                self.process_filing(cik_ticker, year, filing_type, items)
        
        # Generate and return report
        return self.logger.generate_report()


def main():
    """Main entry point for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract items from SEC EDGAR filings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all items from Apple's 2023 10-K
  python main.py --ticker AAPL --filing 10-K --year 2023
  
  # Extract specific items from Microsoft's 2022 and 2023 10-K
  python main.py --ticker MSFT --filing 10-K --years 2022 2023 --items 1 1A 7
  
  # Extract from multiple companies
  python main.py --tickers AAPL MSFT GOOGL --filing 10-K --year 2023
  
  # Use CIK instead of ticker
  python main.py --cik 0000320193 --filing 10-K --year 2023
  
  # Download ALL companies for specific years (no ticker/CIK specified)
  python main.py --filing 10-K --years 2023 2024 2025
        """
    )
    
    # Company identifiers (optional - if not provided, downloads all companies)
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--ticker', '--tickers', nargs='+', dest='tickers',
                      help='Stock ticker symbol(s) (if omitted, downloads all companies)')
    group.add_argument('--cik', '--ciks', nargs='+', dest='ciks',
                      help='CIK number(s) (if omitted, downloads all companies)')
    
    # Filing parameters
    parser.add_argument('--filing', '--filings', nargs='+', dest='filings',
                       required=True, choices=['10-K', '10-Q'],
                       help='Filing type(s)')
    parser.add_argument('--year', '--years', nargs='+', dest='years',
                       required=True, help='Year(s) to extract')
    parser.add_argument('--items', nargs='+', dest='items', default=None,
                       help='Item number(s) to extract (omit to extract all items)')
    
    # Directories
    parser.add_argument('--output-dir', default='sec_filings',
                       help='Output directory for filings (default: sec_filings)')
    parser.add_argument('--log-dir', default='logs',
                       help='Log directory (default: logs)')
    
    # Performance
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of worker threads for parallel extraction (default: 4)')
    
    args = parser.parse_args()
    
    # Convert years to integers
    years = [int(y) for y in args.years]
    
    # Validate years
    for year in years:
        if year < 1995 or year > 2026:
            parser.error(f"Year {year} must be between 1995 and 2026")
    
    # Get company identifiers
    companies = args.tickers if args.tickers else args.ciks
    
    # If no companies specified, download all companies from SEC index
    if not companies:
        print("\n" + "="*80)
        print("WARNING: No tickers or CIKs specified - will download ALL companies!")
        print("="*80)
        
        # Estimate filing count
        index_parser = SECIndexParser()
        for filing_type in args.filings:
            estimated_count, quarters = index_parser.estimate_filing_count(filing_type, years)
            print(f"\nEstimated {filing_type} filings: ~{estimated_count:,}")
            print(f"Years: {', '.join(map(str, years))}")
            print(f"Quarters to check: {quarters}")
        
        print("\nThis may take several hours and require significant storage.")
        print("Rate limiting: 10 requests/second (SEC requirement)")
        
        # Confirmation prompt
        response = input("\nDo you want to continue? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Operation cancelled.")
            sys.exit(0)
        
        print("\nFetching company list from SEC EDGAR full-index...")
        
        # Get all CIKs for each filing type
        all_ciks = set()
        for filing_type in args.filings:
            print(f"\nScanning {filing_type} filings across {len(years)} year(s)...")
            ciks = index_parser.get_ciks_for_filing(filing_type, years)
            all_ciks.update(ciks)
            print(f"Found {len(ciks)} unique companies filing {filing_type}")
        
        companies = sorted(all_ciks)
        print(f"\nTotal unique companies: {len(companies)}")
        print("\nStarting extraction...\n")
    
    # Create extractor
    extractor = ItemXtractor(base_dir=args.output_dir, log_dir=args.log_dir)
    
    # Run extraction (includes automatic structure extraction)
    extractor.extract(
        cik_tickers=companies,
        filing_types=args.filings,
        years=years,
        items=args.items,
        max_workers=args.workers
    )


if __name__ == "__main__":
    main()
