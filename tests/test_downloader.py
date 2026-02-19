"""
Test the downloader module directly
"""

import sys
import os

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.downloader import SECDownloader

def test_downloader():
    """Test the downloader"""
    print("=" * 60)
    print("Testing SECDownloader")
    print("=" * 60)
    
    downloader = SECDownloader()
    
    try:
        print("\nTest 1: Downloading AAPL 2022 10-K...")
        html_content, extension, identifier = downloader.download_filing("AAPL", "10-K", "2022")
        
        print(f"✓ Success!")
        print(f"  Extension: {extension}")
        print(f"  Identifier: {identifier}")
        print(f"  HTML content length: {len(html_content)} characters")
        print(f"  First 200 chars: {html_content[:200]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_downloader()
    
    if success:
        print("\n" + "=" * 60)
        print("✓ Downloader test passed!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ Downloader test failed!")
        print("=" * 60)
