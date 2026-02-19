"""
Test to examine table structure more carefully
"""
import requests
from bs4 import BeautifulSoup

accession = "0000320193-22-000108"
accession_path = accession.replace('-', '')
cik = '0000320193'
index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_path}/{accession}-index.html"

print(f"Fetching: {index_url}\n")

response = requests.get(index_url, headers={'User-Agent': 'Test test@example.com'})
soup = BeautifulSoup(response.content, 'html.parser')

table = soup.find('table', class_='tableFile')
if table:
    rows = table.find_all('tr')
    
    # Print header
    print("Header row:")
    header_cols = rows[0].find_all('th')
    for i, th in enumerate(header_cols):
        print(f"  Col {i}: {th.text.strip()}")
    
    print("\nFirst few data rows:")
    for idx, row in enumerate(rows[1:6]):  # First 5 data rows
        print(f"\nRow {idx}:")
        cols = row.find_all('td')
        for i, col in enumerate(cols):
            text = col.text.strip()
            if col.find('a'):
                href = col.find('a').get('href', '')
                print(f"  Col {i}: {text} [href: {href}]")
            else:
                print(f"  Col {i}: {text}")
