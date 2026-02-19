"""
Configuration settings for ItemXtractor
"""

import os

# SEC EDGAR API Settings
SEC_BASE_URL = "https://www.sec.gov"
SEC_ARCHIVES_URL = f"{SEC_BASE_URL}/cgi-bin/browse-edgar"
SEC_USER_AGENT = "ItemXtractor/1.0 (Research Tool; yourname@yourdomain.com)"  # IMPORTANT: Update with your email

# File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEC_FILINGS_DIR = os.path.join(BASE_DIR, "sec_filings")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Filing Types
SUPPORTED_FILING_TYPES = ["10-K", "10-Q"]

# Item Mappings for 10-K
ITEMS_10K = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "Selected Financial Data",  # Removed in newer filings
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accounting Fees and Services",
    "15": "Exhibits, Financial Statement Schedules",
    "16": "Form 10-K Summary",
}

# Item Mappings for 10-Q
ITEMS_10Q = {
    "1": "Financial Statements",
    "2": "Management's Discussion and Analysis",
    "3": "Quantitative and Qualitative Disclosures About Market Risk",
    "4": "Controls and Procedures",
}

# Request Settings
REQUEST_TIMEOUT = 30  # seconds
REQUEST_DELAY = 0.1  # SEC recommends no more than 10 requests per second

# Logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
