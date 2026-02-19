"""
Logger for tracking extraction operations
"""

import logging
import json
import csv
import os
import threading
from datetime import datetime
from typing import Dict, Any, List
from config import LOGS_DIR, LOG_FORMAT, LOG_DATE_FORMAT, ITEMS_10K, ITEMS_10Q


class ExtractionLogger:
    """Handles logging for SEC filing extraction operations"""
    
    def __init__(self, log_dir: str = LOGS_DIR):
        """
        Initialize ExtractionLogger
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger('ItemXtractor')
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # CSV extraction log for concise tracking
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_log_file = os.path.join(log_dir, f"extraction_{timestamp}.csv")
        self.csv_log_written_header = False
        
        # Session data for report generation
        self.session_data = {
            'start_time': datetime.now().isoformat(),
            'parameters': {},
            'filings': []
        }
        self.lock = threading.Lock()
    
    def set_parameters(self, **kwargs) -> None:
        """
        Set execution parameters
        
        Args:
            **kwargs: Parameter key-value pairs
        """
        with self.lock:
            self.session_data['parameters'] = kwargs
            self.logger.info(f"Parameters: {json.dumps(kwargs, indent=2)}")
    
    def log_filing_start(self, cik_ticker: str, year: str, filing_type: str) -> Dict[str, Any]:
        """
        Log the start of a filing extraction
        
        Args:
            cik_ticker: CIK or ticker
            year: Filing year
            filing_type: Type of filing
            
        Returns:
            Filing record dictionary
        """
        with self.lock:
            filing_record = {
                'cik_ticker': cik_ticker,
                'year': year,
                'filing_type': filing_type,
                'start_time': datetime.now().isoformat(),
                'downloaded': False,
                'skipped_download': False,
                'toc_found': False,
                'items_extracted': [],
                'errors': [],
                'status': 'in_progress'
            }
            self.session_data['filings'].append(filing_record)
            self.logger.info(f"Starting extraction for {cik_ticker} {filing_type} {year}")
            return filing_record
    
    def log_download(self, filing_record: Dict[str, Any], downloaded: bool, 
                    skipped: bool = False, error: str = None) -> None:
        """
        Log download status
        
        Args:
            filing_record: Filing record dictionary
            downloaded: Whether file was successfully downloaded
            skipped: Whether download was skipped (file already exists)
            error: Error message if download failed
        """
        with self.lock:
            filing_record['downloaded'] = downloaded
            filing_record['skipped_download'] = skipped
            
            if error:
                filing_record['errors'].append(f"Download error: {error}")
                self.logger.error(f"Download failed for {filing_record['cik_ticker']} "
                                f"{filing_record['filing_type']} {filing_record['year']}: {error}")
            elif skipped:
                self.logger.info(f"Download skipped (file exists) for {filing_record['cik_ticker']} "
                               f"{filing_record['filing_type']} {filing_record['year']}")
            else:
                self.logger.info(f"Downloaded {filing_record['cik_ticker']} "
                               f"{filing_record['filing_type']} {filing_record['year']}")
    
    def log_toc_detection(self, filing_record: Dict[str, Any], found: bool, 
                         error: str = None) -> None:
        """
        Log Table of Contents detection
        
        Args:
            filing_record: Filing record dictionary
            found: Whether TOC was found
            error: Error message if detection failed
        """
        with self.lock:
            filing_record['toc_found'] = found
            
            if error:
                filing_record['errors'].append(f"TOC detection error: {error}")
                self.logger.error(f"TOC detection failed: {error}")
            elif not found:
                self.logger.warning(f"No TOC found in {filing_record['cik_ticker']} "
                                  f"{filing_record['filing_type']} {filing_record['year']}")
            else:
                self.logger.info(f"TOC found in {filing_record['cik_ticker']} "
                               f"{filing_record['filing_type']} {filing_record['year']}")
    
    def log_item_extraction(self, filing_record: Dict[str, Any], item_number: str, 
                           success: bool, error: str = None) -> None:
        """
        Log item extraction
        
        Args:
            filing_record: Filing record dictionary
            item_number: Item number extracted
            success: Whether extraction was successful
            error: Error message if extraction failed
        """
        with self.lock:
            if success:
                filing_record['items_extracted'].append(item_number)
                self.logger.info(f"Extracted Item {item_number} from {filing_record['cik_ticker']} "
                               f"{filing_record['filing_type']} {filing_record['year']}")
            else:
                error_msg = f"Item {item_number} extraction error: {error}"
                filing_record['errors'].append(error_msg)
                self.logger.error(error_msg)
    
    def log_filing_complete(self, filing_record: Dict[str, Any]) -> None:
        """
        Mark filing processing as complete
        
        Args:
            filing_record: Filing record dictionary
        """
        with self.lock:
            filing_record['end_time'] = datetime.now().isoformat()
            filing_record['status'] = 'completed'
            
            # Calculate duration
            start = datetime.fromisoformat(filing_record['start_time'])
            end = datetime.fromisoformat(filing_record['end_time'])
            duration = (end - start).total_seconds()
            filing_record['duration_seconds'] = duration
            
            self.logger.info(f"Completed {filing_record['cik_ticker']} "
                           f"{filing_record['filing_type']} {filing_record['year']} "
                           f"in {duration:.2f}s - Items: {filing_record['items_extracted']}")
            
            # Write to CSV extraction log
            self._write_csv_log_entry(filing_record)
    
    def _write_csv_log_entry(self, filing_record: Dict[str, Any]) -> None:
        """
        Write a filing entry to the CSV extraction log
        
        Args:
            filing_record: Filing record dictionary
        """
        try:
            # Get all item columns
            all_10k_items = list(ITEMS_10K.keys())
            all_10q_items = list(ITEMS_10Q.keys())
            
            # Write header if first entry
            if not self.csv_log_written_header:
                with open(self.csv_log_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # Write start time
                    start_time = datetime.fromisoformat(self.session_data['start_time'])
                    writer.writerow([f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}"])
                    
                    # Write column headers
                    headers = ['Ticker', 'Year', 'Filing Type', 'Download', 'TOC']
                    headers.extend([f'10-K Item {item}' for item in all_10k_items])
                    headers.extend([f'10-Q Item {item}' for item in all_10q_items])
                    headers.append('Runtime (sec)')
                    writer.writerow(headers)
                self.csv_log_written_header = True
            
            # Write data row
            with open(self.csv_log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                row = [
                    filing_record['cik_ticker'],
                    filing_record['year'],
                    filing_record['filing_type'],
                    'O' if filing_record['downloaded'] or filing_record['skipped_download'] else 'X',
                    'O' if filing_record['toc_found'] else 'X'
                ]
                
                # Add O/X for each 10-K item
                filing_items = set(filing_record['items_extracted'])
                for item in all_10k_items:
                    if filing_record['filing_type'] == '10-K':
                        row.append('O' if item in filing_items else 'X')
                    else:
                        row.append('')  # Empty for 10-Q filings
                
                # Add O/X for each 10-Q item
                for item in all_10q_items:
                    if filing_record['filing_type'] == '10-Q':
                        row.append('O' if item in filing_items else 'X')
                    else:
                        row.append('')  # Empty for 10-K filings
                
                # Add runtime
                runtime = filing_record.get('duration_seconds', 0)
                row.append(f'{runtime:.2f}')
                
                writer.writerow(row)
        except Exception as e:
            self.logger.warning(f"Failed to write CSV log entry: {str(e)}")
    
    def generate_report(self) -> str:
        """
        Generate final execution report in CSV format
        
        Returns:
            CSV report string
        """
        with self.lock:
            self.session_data['end_time'] = datetime.now().isoformat()
            
            # Calculate total duration
            start = datetime.fromisoformat(self.session_data['start_time'])
            end = datetime.fromisoformat(self.session_data['end_time'])
            total_duration = (end - start).total_seconds()
            self.session_data['total_duration_seconds'] = total_duration
            
            # Summary statistics
            total_filings = len(self.session_data['filings'])
            successful_downloads = sum(1 for f in self.session_data['filings'] if f['downloaded'] or f['skipped_download'])
            toc_found = sum(1 for f in self.session_data['filings'] if f['toc_found'])
            total_items = sum(len(f['items_extracted']) for f in self.session_data['filings'])
            
            # Save report to CSV file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = os.path.join(self.log_dir, f"report_{timestamp}.csv")
            
            # Get all item columns (10-K items followed by 10-Q items)
            all_10k_items = list(ITEMS_10K.keys())
            all_10q_items = list(ITEMS_10Q.keys())
            
            with open(report_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Line 1: Start time
                writer.writerow([f"Start Time: {start.strftime('%Y-%m-%d %H:%M:%S')}"])
                
                # Line 2: Column headers
                headers = ['Ticker', 'Year', 'Filing Type']
                headers.extend([f'10-K Item {item}' for item in all_10k_items])
                headers.extend([f'10-Q Item {item}' for item in all_10q_items])
                headers.append('Runtime (sec)')
                writer.writerow(headers)
                
                # Data rows: One row per filing
                for filing in self.session_data['filings']:
                    row = [
                        filing['cik_ticker'],
                        filing['year'],
                        filing['filing_type']
                    ]
                    
                    # Add O/X for each 10-K item
                    filing_items = set(filing['items_extracted'])
                    for item in all_10k_items:
                        if filing['filing_type'] == '10-K':
                            row.append('O' if item in filing_items else 'X')
                        else:
                            row.append('')  # Empty for 10-Q filings
                    
                    # Add O/X for each 10-Q item
                    for item in all_10q_items:
                        if filing['filing_type'] == '10-Q':
                            row.append('O' if item in filing_items else 'X')
                        else:
                            row.append('')  # Empty for 10-K filings
                    
                    # Add runtime
                    runtime = filing.get('duration_seconds', 0)
                    row.append(f'{runtime:.2f}')
                    
                    writer.writerow(row)
                
                # Empty line before summary
                writer.writerow([])
                
                # Summary section
                writer.writerow([f"End Time: {end.strftime('%Y-%m-%d %H:%M:%S')}"])
                writer.writerow([f"Total Runtime (sec): {total_duration:.2f}"])
                writer.writerow([])
                writer.writerow(['Summary'])
                writer.writerow(['Total Filings', total_filings])
                writer.writerow(['Successful Downloads', successful_downloads])
                writer.writerow(['TOC Found', toc_found])
                writer.writerow(['Total Items Extracted', total_items])
            
            self.logger.info(f"Execution Report:")
            self.logger.info(f"  Total Duration: {total_duration:.2f}s")
            self.logger.info(f"  Total Filings: {total_filings}")
            self.logger.info(f"  Successful Downloads: {successful_downloads}")
            self.logger.info(f"  TOC Found: {toc_found}")
            self.logger.info(f"  Total Items Extracted: {total_items}")
            self.logger.info(f"  Report saved to: {report_file}")
            
            return f"Report saved to: {report_file}"
    
    def info(self, message: str) -> None:
        """Log info message"""
        with self.lock:
            self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message"""
        with self.lock:
            self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message"""
        with self.lock:
            self.logger.error(message)
