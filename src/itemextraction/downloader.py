"""
SEC EDGAR Downloader - Fetches filings from SEC EDGAR
"""

import requests
import time
import re
from typing import Optional, Tuple
from bs4 import BeautifulSoup
from script.config import (
    SEC_BASE_URL, SEC_USER_AGENT, REQUEST_TIMEOUT, REQUEST_DELAY
)


class SECDownloader:
    """Downloads SEC filings from EDGAR"""
    
    def __init__(self, user_agent: str = SEC_USER_AGENT):
        """
        Initialize SECDownloader
        
        Args:
            user_agent: User agent string for SEC requests
        """
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': user_agent})
    
    def _get_cik_from_ticker(self, ticker: str) -> Optional[str]:
        """
        Get CIK number from ticker symbol
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            CIK number (padded to 10 digits) or None if not found
        """
        try:
            # Use SEC's company tickers JSON endpoint
            url = f"{SEC_BASE_URL}/files/company_tickers.json"
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            ticker_upper = ticker.upper()
            
            for entry in data.values():
                if entry.get('ticker', '').upper() == ticker_upper:
                    cik = str(entry['cik_str']).zfill(10)
                    return cik
            
            return None
        except Exception as e:
            raise Exception(f"Failed to resolve ticker {ticker}: {str(e)}")
    
    def _normalize_cik(self, cik_or_ticker: str) -> Tuple[str, str]:
        """
        Normalize CIK or ticker to CIK number
        
        Args:
            cik_or_ticker: CIK number or ticker symbol
            
        Returns:
            Tuple of (CIK number padded to 10 digits, original identifier)
        """
        # Check if it's already a CIK (numeric)
        if cik_or_ticker.isdigit():
            return cik_or_ticker.zfill(10), cik_or_ticker
        
        # Otherwise, treat as ticker and resolve
        cik = self._get_cik_from_ticker(cik_or_ticker)
        if cik is None:
            raise ValueError(f"Could not resolve ticker: {cik_or_ticker}")
        
        return cik, cik_or_ticker
    
    def get_cik(self, cik_or_ticker: str) -> str:
        """
        Get CIK number from ticker or CIK
        
        Args:
            cik_or_ticker: CIK number or ticker symbol
            
        Returns:
            CIK number padded to 10 digits
        """
        cik, _ = self._normalize_cik(cik_or_ticker)
        return cik
    
    def _get_filing_url(self, cik: str, filing_type: str, year: str) -> Optional[str]:
        """
        Get the URL of the filing document
        
        Args:
            cik: CIK number (10 digits)
            filing_type: Type of filing (10-K or 10-Q)
            year: Filing year
            
        Returns:
            URL of the filing HTML document or None if not found
        """
        try:
            # Search for filings using JSON endpoint (more reliable)
            search_url = f"{SEC_BASE_URL}/files/company_tickers.json"
            
            # First, get the company's recent filings
            browse_url = f"{SEC_BASE_URL}/cgi-bin/browse-edgar"
            params = {
                'action': 'getcompany',
                'CIK': cik,
                'type': filing_type,
                'owner': 'exclude',
                'output': 'atom',
                'count': '100'
            }
            
            time.sleep(REQUEST_DELAY)
            response = self.session.get(browse_url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            # Parse XML/Atom response
            soup = BeautifulSoup(response.content, 'lxml-xml')
            
            # Find entries (filings)
            entries = soup.find_all('entry')
            if not entries:
                raise Exception(f"No filings found for CIK {cik}")
            
            # Find the filing from the specified year
            # Collect all filings from the year, then sort to prioritize non-amendments
            filings_from_year = []
            
            for entry in entries:
                filing_date_elem = entry.find('filing-date')
                if not filing_date_elem:
                    continue
                    
                filing_date = filing_date_elem.text
                
                # Check if filing is from the specified year
                if filing_date.startswith(year):
                    # Get the accession number
                    accession_elem = entry.find('accession-number')
                    if not accession_elem:
                        continue
                    
                    accession_formatted = accession_elem.text  # Format: 0001193125-23-123456
                    accession_path = accession_formatted.replace('-', '')  # Remove dashes for path
                    
                    filings_from_year.append((accession_formatted, accession_path, filing_date))
            
            # Sort filings to prioritize main filing over amendments (10-K before 10-K/A)
            # Earlier amendments have higher accession numbers, so non-amendments typically come first
            # Process in order but prefer non-amendments
            for accession_formatted, accession_path, filing_date in filings_from_year:
                    
                    # Construct direct URL to filing documents
                    cik_padded = cik.zfill(10)
                    
                    # The filing documents are typically at:
                    # https://www.sec.gov/Archives/edgar/{CIK}/{accession_no_dashes}/{accession_no_with_dashes}-index.html
                    doc_index_url = f"{SEC_BASE_URL}/Archives/edgar/data/{cik_padded}/{accession_path}/{accession_formatted}-index.html"
                    
                    time.sleep(REQUEST_DELAY)
                    try:
                        doc_response = self.session.get(doc_index_url, timeout=REQUEST_TIMEOUT)
                        
                        if doc_response.status_code == 200:
                            # Parse the index to find the main document
                            doc_soup = BeautifulSoup(doc_response.content, 'html.parser')
                            
                            # Look for the main filing document (usually .htm or .html)
                            table = doc_soup.find('table', class_='tableFile')
                            if table:
                                rows = table.find_all('tr')[1:]  # Skip header
                                
                                # Check if this is an amendment - skip amended filings
                                # Look at document types to see if this is an amendment version
                                is_amendment = False
                                for row in rows:
                                    cols = row.find_all('td')
                                    if len(cols) >= 4:
                                        doc_type = cols[3].text.strip()
                                        # Skip if document type indicates amendment (e.g., "10-K/A", "10-Q/A", etc.)
                                        if '/A' in doc_type:
                                            is_amendment = True
                                            break
                                
                                # If this is an amendment, skip and try the next filing
                                if is_amendment:
                                    continue
                                
                                # The main filing is typically sequence 1 with type matching filing_type
                                # Look for first HTML file in sequence 1 or 2
                                for row in rows:
                                    cols = row.find_all('td')
                                    if len(cols) >= 4:
                                        sequence = cols[0].text.strip()
                                        filename = cols[2].text.strip()
                                        doc_type = cols[3].text.strip()
                                        
                                        # Main filing is usually sequence 1
                                        if sequence == '1' and (filename.endswith('.htm') or filename.endswith('.html') or 'htm' in filename):
                                            link = cols[2].find('a')
                                            if link and link.get('href'):
                                                href = link['href']
                                                # Handle iXBRL viewer links: /ix?doc=/Archives/edgar/...
                                                if '/ix?doc=' in href:
                                                    # Extract the actual document path from the iXBRL viewer URL
                                                    doc_path = href.split('/ix?doc=')[1]
                                                    doc_url = f"{SEC_BASE_URL}{doc_path}"
                                                else:
                                                    doc_url = f"{SEC_BASE_URL}{href}"
                                                return doc_url
                    except:
                        # If index file not found, try alternative approach
                        pass
            
            return None
        except Exception as e:
            raise Exception(f"Failed to get filing URL: {str(e)}")

    
    def download_filing(self, cik_or_ticker: str, filing_type: str, 
                       year: str) -> Tuple[str, str, str]:
        """
        Download a SEC filing
        
        Args:
            cik_or_ticker: CIK number or ticker symbol
            filing_type: Type of filing (10-K or 10-Q)
            year: Filing year
            
        Returns:
            Tuple of (HTML content, file extension, CIK number padded to 10 digits)
            
        Raises:
            Exception: If download fails
        """
        # Normalize CIK
        cik, original_identifier = self._normalize_cik(cik_or_ticker)
        
        # Get filing URL
        filing_url = self._get_filing_url(cik, filing_type, year)
        
        if not filing_url:
            raise Exception(f"No {filing_type} filing found for {original_identifier} in {year}")
        
        # Download the filing
        time.sleep(REQUEST_DELAY)
        response = self.session.get(filing_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        # Determine file extension from URL
        extension = 'html'
        if filing_url.endswith('.htm'):
            extension = 'htm'
        
        return response.text, extension, cik
