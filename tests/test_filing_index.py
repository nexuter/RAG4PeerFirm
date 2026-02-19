"""
Test script to examine EDGAR filing index
"""
import requests
from bs4 import BeautifulSoup

# AAPL 2022 10-K accession number
# Format: ####-##-######
url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193&type=10-K&owner=exclude&output=atom&count=10"

response = requests.get(url, headers={'User-Agent': 'Test test@example.com'})
soup = BeautifulSoup(response.content, 'lxml-xml')

entries = soup.find_all('entry')
for entry in entries:
    filing_date = entry.find('filing-date').text if entry.find('filing-date') else 'N/A'
    accession = entry.find('accession-number').text if entry.find('accession-number') else 'N/A'
    
    if filing_date.startswith('2022'):
        print(f"\nFiling Date: {filing_date}")
        print(f"Accession: {accession}")
        
        # Get the filing index
        accession_path = accession.replace('-', '')
        cik = '0000320193'
        index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_path}/{accession}-index.html"
        
        print(f"Index URL: {index_url}")
        
        index_resp = requests.get(index_url, headers={'User-Agent': 'Test test@example.com'})
        if index_resp.status_code == 200:
            index_soup = BeautifulSoup(index_resp.content, 'html.parser')
            table = index_soup.find('table', class_='tableFile')
            
            if table:
                print("\nDocuments in filing:")
                print(f"{'Seq':<5} {'Document':<40} {'Type':<15} {'Size'}")
                print("-" * 80)
                
                rows = table.find_all('tr')[1:]  # Skip header
                for row in rows[:15]:  # Show first 15
                    cols = row.find_all('td')
                    if len(cols) >= 4:
                        seq = cols[0].text.strip()
                        filename = cols[2].text.strip()
                        doc_type = cols[3].text.strip()
                        size = cols[4].text.strip() if len(cols) > 4 else 'N/A'
                        
                        print(f"{seq:<5} {filename:<40} {doc_type:<15} {size}")
        
        break  # Only check first 2022 filing
