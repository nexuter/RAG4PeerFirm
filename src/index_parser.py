"""
SEC EDGAR Full-Index Parser - Retrieves all companies from quarterly index files
"""

import requests
import re
from typing import List, Set, Dict, Tuple
from config import SEC_BASE_URL, SEC_USER_AGENT, REQUEST_TIMEOUT, REQUEST_DELAY
import time


class SECIndexParser:
    """Parses SEC EDGAR full-index files to get all companies for a filing type"""
    
    def __init__(self, user_agent: str = SEC_USER_AGENT):
        """
        Initialize SECIndexParser
        
        Args:
            user_agent: User agent string for SEC requests
        """
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': user_agent})
    
    def _download_index_file(self, year: int, quarter: int) -> str:
        """
        Download company.idx file for a specific year and quarter
        
        Args:
            year: Year (e.g., 2023)
            quarter: Quarter (1-4)
            
        Returns:
            Content of the index file as string
            
        Raises:
            Exception if download fails
        """
        url = f"{SEC_BASE_URL}/Archives/edgar/full-index/{year}/QTR{quarter}/company.idx"
        
        try:
            time.sleep(REQUEST_DELAY)  # Rate limiting
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.text
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                # Quarter might not be available yet
                return ""
            raise Exception(f"Failed to download index for {year} Q{quarter}: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to download index for {year} Q{quarter}: {str(e)}")
    
    def _parse_index_file(self, content: str, filing_type: str) -> List[Dict[str, str]]:
        """
        Parse company.idx file and extract filings of specific type
        
        Args:
            content: Content of the index file
            filing_type: Filing type to filter (e.g., '10-K')
            
        Returns:
            List of dictionaries with company info
        """
        filings = []
        
        # Skip header lines (usually first 10 lines are headers)
        lines = content.split('\n')
        data_started = False
        
        for line in lines:
            # Header ends with a line of dashes
            if '---' in line:
                data_started = True
                continue
            
            if not data_started or not line.strip():
                continue
            
            # Parse line: Company Name | Form Type | CIK | Date Filed | File Name
            # Lines are fixed-width or pipe-separated (varies by year)
            parts = line.split('|') if '|' in line else None
            
            if parts and len(parts) >= 5:
                # Pipe-separated format
                company_name = parts[0].strip()
                form_type = parts[1].strip()
                cik = parts[2].strip()
                date_filed = parts[3].strip()
                file_name = parts[4].strip()
            else:
                # Fixed-width format (older indices)
                # Approximate column positions based on SEC format
                # Company Name: 0-62, Form Type: 62-74, CIK: 74-86, Date Filed: 86-98, File Name: 98+
                if len(line) < 98:
                    continue
                
                company_name = line[0:62].strip()
                form_type = line[62:74].strip()
                cik = line[74:86].strip()
                date_filed = line[86:98].strip()
                file_name = line[98:].strip()
            
            # Filter by filing type
            if form_type == filing_type:
                # Normalize CIK (remove leading zeros for consistency)
                cik_normalized = cik.lstrip('0') or '0'
                
                filings.append({
                    'company_name': company_name,
                    'form_type': form_type,
                    'cik': cik_normalized,
                    'cik_padded': cik.zfill(10),
                    'date_filed': date_filed,
                    'file_name': file_name
                })
        
        return filings
    
    def get_all_companies_for_filing(self, filing_type: str, years: List[int]) -> List[Dict[str, str]]:
        """
        Get all companies that filed a specific form type across multiple years
        
        Args:
            filing_type: Filing type (e.g., '10-K', '10-Q')
            years: List of years to search
            
        Returns:
            List of unique filings with company info, sorted by date
        """
        all_filings = []
        seen_combinations = set()  # Track (CIK, year) to avoid duplicates
        
        for year in years:
            for quarter in range(1, 5):  # Q1-Q4
                try:
                    # Download index file
                    content = self._download_index_file(year, quarter)
                    
                    if not content:
                        # Quarter not available (e.g., future quarter)
                        continue
                    
                    # Parse and filter
                    filings = self._parse_index_file(content, filing_type)
                    
                    # Add to results, avoiding duplicates
                    for filing in filings:
                        # Extract year from date_filed (format: YYYY-MM-DD)
                        filed_year = filing['date_filed'][:4] if filing['date_filed'] else str(year)
                        
                        # Create unique key
                        key = (filing['cik'], filed_year)
                        
                        if key not in seen_combinations:
                            seen_combinations.add(key)
                            all_filings.append(filing)
                    
                except Exception as e:
                    # Log error but continue with other quarters
                    print(f"Warning: Failed to process {year} Q{quarter}: {str(e)}")
                    continue
        
        # Sort by date filed
        all_filings.sort(key=lambda x: x['date_filed'], reverse=True)
        
        return all_filings
    
    def get_ciks_for_filing(self, filing_type: str, years: List[int]) -> List[str]:
        """
        Get list of unique CIKs that filed a specific form type
        
        Args:
            filing_type: Filing type (e.g., '10-K', '10-Q')
            years: List of years to search
            
        Returns:
            List of unique CIK numbers (padded to 10 digits)
        """
        filings = self.get_all_companies_for_filing(filing_type, years)
        
        # Extract unique CIKs
        ciks = sorted(set(filing['cik_padded'] for filing in filings))
        
        return ciks
    
    def estimate_filing_count(self, filing_type: str, years: List[int]) -> Tuple[int, int]:
        """
        Estimate the number of filings without downloading full indices
        
        Args:
            filing_type: Filing type (e.g., '10-K', '10-Q')
            years: List of years
            
        Returns:
            Tuple of (estimated_count, quarters_checked)
        """
        # For 10-K: typically 4000-5000 per year
        # For 10-Q: typically 12000-15000 per year (3 quarters × 4000-5000 companies)
        
        estimated_counts = {
            '10-K': 4500,  # Average per year
            '10-Q': 13500  # Average per year (3 quarters)
        }
        
        base_count = estimated_counts.get(filing_type, 5000)
        total_estimate = base_count * len(years)
        quarters = len(years) * 4
        
        return total_estimate, quarters
