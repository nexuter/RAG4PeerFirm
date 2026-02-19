"""
Add structure extraction integration to ItemXtractor

This patch adds:
1. Import StructureExtractor
2. Initialize structure_extractor in __init__
3. Add extract_structure_from_items() method
4. Add _extract_item_structure() worker method
"""

# Changes needed in main.py:

# 1. Add import (after line 15):
from src.structure_extractor import StructureExtractor

# 2. In __init__ (after line 35):
self.structure_extractor = StructureExtractor()

# 3. Add new method after _extract_and_save_item():
def _extract_item_structure(self, cik_ticker: str, year: str, filing_type: str,
                            item_number: str) -> bool:
    """
    Extract hierarchical structure from an item
    
    Args:
        cik_ticker: CIK or ticker symbol
        year: Filing year
        filing_type: Type of filing
        item_number: Item number
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load item JSON
        item_path = self.file_manager.get_item_path(
            cik_ticker, year, filing_type, item_number
        )
        
        if not self.file_manager.file_exists(item_path):
            self.logger.warning(
                f"Item file not found for structure extraction: {item_path}"
            )
            return False
        
        # Load item data
        item_data = self.file_manager.load_json(item_path)
        item_html = item_data.get('html_content', '')
        
        if not item_html:
            self.logger.warning(
                f"No HTML content in item {item_number} for {cik_ticker} {filing_type} {year}"
            )
            return False
        
        # Extract structure
        structure = self.structure_extractor.extract_structure(item_html)
        
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
                f"Extracted structure for Item {item_number} from {cik_ticker} {filing_type} {year} "
                f"({len(structure)} elements)"
            )
        
        return True
        
    except Exception as e:
        with self.logger_lock:
            self.logger.error(
                f"Failed to extract structure for Item {item_number}: {str(e)}"
            )
        return False

def extract_structures(self, cik_tickers: Union[str, List[str]],
                      filing_types: Union[str, List[str]],
                      years: Union[str, int, List[Union[str, int]]],
                      items: Optional[List[str]] = None,
                      max_workers: int = 4) -> str:
    """
    Extract hierarchical structure from already extracted items
    
    Args:
        cik_tickers: CIK number(s) or ticker symbol(s)
        filing_types: Filing type(s) (10-K, 10-Q)
        years: Year(s) to extract
        items: List of item numbers to extract (None = extract all)
        max_workers: Number of worker threads for parallel extraction
        
    Returns:
        Summary message
    """
    # Normalize inputs
    if isinstance(cik_tickers, str):
        cik_tickers = [cik_tickers]
    if isinstance(filing_types, str):
        filing_types = [filing_types]
    if isinstance(years, (str, int)):
        years = [str(years)]
    else:
        years = [str(year) for year in years]
    
    self.logger.info("Starting structure extraction...")
    
    # Build tasks
    tasks = []
    for cik_ticker in cik_tickers:
        for filing_type in filing_types:
            for year in years:
                # Get items to process
                if items is None:
                    available_items = self._get_available_items(filing_type)
                else:
                    available_items = items
                
                for item_number in available_items:
                    tasks.append((cik_ticker, year, filing_type, item_number))
    
    # Process in parallel
    successful = 0
    failed = 0
    
    if len(tasks) > 1 and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="structure-worker") as executor:
            futures = []
            for cik_ticker, year, filing_type, item_number in tasks:
                future = executor.submit(
                    self._extract_item_structure,
                    cik_ticker, year, filing_type, item_number
                )
                futures.append(future)
            
            for future in futures:
                if future.result():
                    successful += 1
                else:
                    failed += 1
    else:
        for cik_ticker, year, filing_type, item_number in tasks:
            if self._extract_item_structure(cik_ticker, year, filing_type, item_number):
                successful += 1
            else:
                failed += 1
    
    summary = f"Structure extraction complete: {successful} successful, {failed} failed"
    self.logger.info(summary)
    return summary
