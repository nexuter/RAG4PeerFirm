"""
Filing Analysis Statistics Generator
Analyzes downloaded SEC filings and generates comprehensive descriptive statistics report
"""

import json
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import statistics
import sys
import re

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from config import ITEMS_10K, ITEMS_10Q

class FilingAnalyzer:
    """Analyzes SEC filings and generates statistics report"""
    
    def __init__(self, filings_folder):
        self.filings_folder = Path(filings_folder)
        self.stats_by_year = defaultdict(lambda: {
            'filings': defaultdict(int),
            'items': defaultdict(lambda: {
                'success': 0,
                'failed': 0,
                'headings': [],
                'bodies': [],
                'elements': [],
                'depths': [],
                'title': None
            }),
            'errors': []
        })
        self.extra_items = defaultdict(set)  # item_num -> set of (cik, title)
        self.report = []
        
    def analyze(self):
        """Run complete analysis"""
        print("Scanning filings folder...")
        self._scan_filings()
        
        print("Analyzing structure files...")
        self._analyze_structures()
        
        print("Generating report...")
        self._generate_report()
        
        return self._write_report()
    
    def _scan_filings(self):
        """Scan filings folder and collect statistics"""
        if not self.filings_folder.exists():
            raise Exception(f"Folder not found: {self.filings_folder}")
        
        # Walk through folder structure: {cik}/{year}/{filing_type}/
        for cik_folder in self.filings_folder.iterdir():
            if not cik_folder.is_dir() or cik_folder.name.startswith('.'):
                continue
            
            cik = cik_folder.name
            
            for year_folder in cik_folder.iterdir():
                if not year_folder.is_dir():
                    continue
                
                try:
                    year = int(year_folder.name)
                except ValueError:
                    continue
                
                for filing_folder in year_folder.iterdir():
                    if not filing_folder.is_dir():
                        continue
                    
                    filing_type = filing_folder.name
                    self.stats_by_year[year]['filings'][filing_type] += 1
                    
                    # Scan items folder
                    items_folder = filing_folder / 'items'
                    if items_folder.exists():
                        self._analyze_items_folder(items_folder, year, filing_type, cik)
    
    def _analyze_items_folder(self, items_folder, year, filing_type, cik):
        """Analyze items in a filing folder and extract TOC info"""
        for item_file in items_folder.glob('*_item*.json'):
            if item_file.name.endswith('_xtr.json'):
                continue  # Skip structure files for now
            
            # Extract item number from filename
            parts = item_file.stem.split('_item')
            if len(parts) < 2:
                continue
            
            item_num = parts[-1]
            
            # Try to get item title from JSON (from TOC parsing)
            title = None
            try:
                with open(item_file, 'r', encoding='utf-8', errors='ignore') as f:
                    item_data = json.load(f)
                    title = item_data.get('item_title') or item_data.get('item_number') or item_num
                    
                self.stats_by_year[year]['items'][item_num]['success'] += 1
                self.stats_by_year[year]['items'][item_num]['title'] = title
                
                # Track extra items (not in ITEMS_10K)
                if item_num not in ITEMS_10K:
                    self.extra_items[item_num].add((cik, title))
                    
            except Exception as e:
                self.stats_by_year[year]['items'][item_num]['failed'] += 1
                self.stats_by_year[year]['errors'].append({
                    'cik': cik,
                    'item': item_num,
                    'error': str(e)
                })
    
    def _analyze_structures(self):
        """Analyze structure files (*_xtr.json)"""
        for cik_folder in self.filings_folder.iterdir():
            if not cik_folder.is_dir() or cik_folder.name.startswith('.'):
                continue
            
            for year_folder in cik_folder.iterdir():
                if not year_folder.is_dir():
                    continue
                
                try:
                    year = int(year_folder.name)
                except ValueError:
                    continue
                
                for filing_folder in year_folder.iterdir():
                    if not filing_folder.is_dir():
                        continue
                    
                    items_folder = filing_folder / 'items'
                    if not items_folder.exists():
                        continue
                    
                    for xtr_file in items_folder.glob('*_xtr.json'):
                        self._analyze_structure_file(xtr_file, year)
    
    def _analyze_structure_file(self, xtr_file, year):
        """Analyze a single structure file"""
        try:
            with open(xtr_file, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
            
            item_num = data.get('item_number', 'Unknown')
            structure = data.get('structure', [])
            
            if structure:
                headings = self._count_headings(structure)
                bodies = self._count_bodies(structure)
                depth = self._max_depth(structure)
                
                self.stats_by_year[year]['items'][item_num]['headings'].append(headings)
                self.stats_by_year[year]['items'][item_num]['bodies'].append(bodies)
                self.stats_by_year[year]['items'][item_num]['elements'].append(len(self._flatten_structure(structure)))
                self.stats_by_year[year]['items'][item_num]['depths'].append(depth)
        except Exception as e:
            pass
    
    def _count_headings(self, structure, bold_only=False):
        """Count headings in structure"""
        count = 0
        for elem in structure:
            if elem.get('type') in ['heading', 'bold_heading']:
                if not bold_only or elem.get('type') == 'bold_heading':
                    count += 1
            if 'children' in elem:
                count += self._count_headings(elem['children'], bold_only)
        return count
    
    def _count_bodies(self, structure):
        """Count non-empty bodies in structure"""
        count = 0
        for elem in structure:
            if elem.get('body', '').strip():
                count += 1
            if 'children' in elem:
                count += self._count_bodies(elem['children'])
        return count
    
    def _max_depth(self, structure, depth=0):
        """Get maximum nesting depth"""
        max_d = depth
        for elem in structure:
            if 'children' in elem:
                max_d = max(max_d, self._max_depth(elem['children'], depth + 1))
        return max_d
    
    def _flatten_structure(self, structure):
        """Flatten structure to count all elements"""
        elements = []
        for elem in structure:
            elements.append(elem)
            if 'children' in elem:
                elements.extend(self._flatten_structure(elem['children']))
        return elements
    
    def _generate_report(self):
        """Generate markdown report"""
        self.report = []
        
        # Header
        self.report.append("# Filing Analysis Report")
        self.report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.report.append("")
        
        # Executive Summary
        self._add_executive_summary()
        
        # Item Extraction Analysis (sorted by ITEMS_10K)
        self._add_item_extraction_sorted()
        
        # Structure Extraction Analysis
        self._add_structure_analysis()
        
        # Headings & Bodies Analysis
        self._add_headings_bodies_analysis()
        
        # Item Statistics Summary
        self._add_item_statistics_summary()
        
        # Extra Items Investigation
        if self.extra_items:
            self._add_extra_items_section()
        
        # Error Report
        self._add_error_report()
        
        # Conclusions
        self._add_conclusions()
    
    def _add_executive_summary(self):
        """Add executive summary section"""
        self.report.append("## 1. Executive Summary")
        self.report.append("")
        
        total_filings = sum(sum(v['filings'].values()) 
                           for v in self.stats_by_year.values())
        years = sorted(self.stats_by_year.keys())
        
        self.report.append(f"- **Total Filings Analyzed**: {total_filings:,}")
        self.report.append(f"- **Years Covered**: {years[0]}-{years[-1]}")
        self.report.append(f"- **Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.report.append("")
        
        # Year breakdown
        self.report.append("### Filings by Year")
        self.report.append("")
        for year in years:
            total = sum(self.stats_by_year[year]['filings'].values())
            self.report.append(f"- **{year}**: {total:,} filings")
        self.report.append("")
    
    def _get_item_order(self):
        """Get items sorted by ITEMS_10K order, with alpha sub-items grouped under parents"""
        result = []
        
        # Collect all items seen in data
        items_present = set()
        for year in self.stats_by_year.keys():
            items_present.update(self.stats_by_year[year]['items'].keys())
        
        added = set()
        
        # Add items in ITEMS_10K order
        for item_key in ITEMS_10K.keys():
            result.append(item_key)
            added.add(item_key)
            
            # Group alpha sub-items (e.g., 8A, 8B) under numeric parent
            if item_key.isdigit():
                prefix = item_key
                sub_items = sorted(
                    item for item in items_present
                    if re.match(rf"^{prefix}[A-Z]+$", item) and item not in ITEMS_10K
                )
                for sub in sub_items:
                    if sub not in added:
                        result.append(sub)
                        added.add(sub)
        
        # Append remaining extra items not grouped (e.g., 1101, 401)
        for item in sorted(items_present):
            if item not in added:
                result.append(item)
                added.add(item)
        
        return result

    def _get_item_title(self, item_num):
        """Get item title from any year"""
        for year in sorted(self.stats_by_year.keys()):
            title = self.stats_by_year[year]['items'][item_num].get('title')
            if title:
                return title
        return None

    def _get_item_description(self, item_num):
        """Get description for item based on mapping or TOC title"""
        if item_num in ITEMS_10K:
            return ITEMS_10K[item_num]
        title = self._get_item_title(item_num)
        return title or "Unknown/Extra Item"
    
    def _add_item_extraction_sorted(self):
        """Add item extraction analysis sorted by ITEMS_10K"""
        self.report.append("## 2. Item Extraction Analysis (Sorted by ITEMS_10K)")
        self.report.append("")
        
        years = sorted(self.stats_by_year.keys())
        item_order = self._get_item_order()
        
        # Build header with year columns
        header = "| Item | Description | "
        for year in years:
            header += f"{year} Success | "
        header += "Overall Success |"
        self.report.append(header)
        
        # Build separator
        sep = "|" + "|".join(["-" * 18] * (len(years) + 3)) + "|"
        self.report.append(sep)
        
        # Add rows for each item in ITEMS_10K order
        for item in item_order:
            description = self._get_item_description(item)
            
            # Determine if this is a sub-item (alpha suffix)
            is_sub_item = bool(re.match(r"^\d+[A-Z]+$", item)) and item not in ITEMS_10K
            prefix = "  └─ " if is_sub_item else ""
            
            row = f"| {prefix}Item {item} | {description} | "
            
            for year in years:
                success = self.stats_by_year[year]['items'][item]['success']
                row += f"{success:,} | "
            
            # Overall count
            total_success = sum(self.stats_by_year[y]['items'][item]['success'] 
                               for y in years)
            row += f"{total_success:,} |"
            
            self.report.append(row)
        
        self.report.append("")
    
    def _add_structure_analysis(self):
        """Add structure extraction analysis with year columns"""
        self.report.append("## 3. Structure Extraction Analysis")
        self.report.append("")
        
        years = sorted(self.stats_by_year.keys())
        item_order = self._get_item_order()
        
        # Build header
        header = "| Item | "
        for year in years:
            header += f"{year} Avg Depth | {year} Max Depth | "
        self.report.append(header)
        
        # Build separator
        sep = "|" + "|".join(["-" * 14] * (len(years) * 2 + 1)) + "|"
        self.report.append(sep)
        
        # Add rows
        for item in item_order:
            row = f"| Item {item} | "
            
            for year in years:
                depths = self.stats_by_year[year]['items'][item]['depths']
                
                if depths:
                    avg_depth = statistics.mean(depths)
                    max_depth = max(depths)
                    row += f"{avg_depth:.1f} | {max_depth} | "
                else:
                    row += "- | - | "
            
            self.report.append(row)
        
        self.report.append("")
    
    def _add_headings_bodies_analysis(self):
        """Add headings and bodies analysis with year columns"""
        self.report.append("## 4. Headings & Bodies Analysis by Item (Year Comparison)")
        self.report.append("")
        
        years = sorted(self.stats_by_year.keys())
        item_order = self._get_item_order()
        
        # Build header
        header = "| Item | "
        for year in years:
            header += f"{year} Avg Headings | {year} Avg Bodies | "
        self.report.append(header)
        
        # Build separator
        sep = "|" + "|".join(["-" * 18] * (len(years) * 2 + 1)) + "|"
        self.report.append(sep)
        
        # Add rows
        for item in item_order:
            row = f"| Item {item} | "
            
            for year in years:
                headings = self.stats_by_year[year]['items'][item]['headings']
                bodies = self.stats_by_year[year]['items'][item]['bodies']
                
                if headings and bodies:
                    avg_h = statistics.mean(headings)
                    avg_b = statistics.mean(bodies)
                    row += f"{avg_h:.1f} | {avg_b:.1f} | "
                else:
                    row += "- | - | "
            
            self.report.append(row)
        
        self.report.append("")
    
    def _add_item_statistics_summary(self):
        """Add item statistics summary with year comparison"""
        self.report.append("## 5. Item Statistics Summary (Year Comparison)")
        self.report.append("")
        
        years = sorted(self.stats_by_year.keys())
        item_order = self._get_item_order()
        
        # Build header
        header = "| Item | "
        for year in years:
            header += f"{year} Count | {year} Avg Depth | {year} Avg Headings | {year} Avg Bodies | "
        self.report.append(header)
        
        # Build separator
        sep = "|" + "|".join(["-" * 16] * (len(years) * 4 + 1)) + "|"
        self.report.append(sep)
        
        # Add rows
        for item in item_order:
            row = f"| Item {item} | "
            for year in years:
                item_stats = self.stats_by_year[year]['items'][item]
                count = item_stats['success']
                depths = item_stats['depths']
                headings = item_stats['headings']
                bodies = item_stats['bodies']
                
                if count > 0:
                    avg_depth = statistics.mean(depths) if depths else 0
                    avg_h = statistics.mean(headings) if headings else 0
                    avg_b = statistics.mean(bodies) if bodies else 0
                    row += f"{count:,} | {avg_depth:.1f} | {avg_h:.0f} | {avg_b:.0f} | "
                else:
                    row += "- | - | - | - | "
            self.report.append(row)
        
        self.report.append("")
    
    def _add_extra_items_section(self):
        """Add section for extra items not in ITEMS_10K"""
        self.report.append("## 6. Extra/Unknown Items Investigation")
        self.report.append("")
        self.report.append(f"Found {len(self.extra_items)} item types not in ITEMS_10K mapping:")
        self.report.append("")
        
        years = sorted(self.stats_by_year.keys())
        
        for item_num in sorted(self.extra_items.keys()):
            cik_title_set = self.extra_items[item_num]
            count = len(cik_title_set)
            
            # Count by year
            year_counts = {}
            for year in years:
                year_counts[year] = self.stats_by_year[year]['items'][item_num]['success']
            
            self.report.append(f"### Item {item_num} (Found in {count} filing(s))")
            self.report.append("")
            
            # Year comparison table
            header = "| Year | Count |"
            self.report.append(header)
            self.report.append("|------|-------|")
            for year in years:
                self.report.append(f"| {year} | {year_counts[year]:,} |")
            self.report.append("")
            
            # Sample CIK/title examples
            self.report.append("| CIK | Title/Description |")
            self.report.append("|-----|-------------------|")
            
            for cik, title in list(cik_title_set)[:5]:
                title_display = str(title)[:60] if title else "Unknown"
                self.report.append(f"| {cik} | {title_display} |")
            
            if count > 5:
                self.report.append(f"| ... | +{count - 5} more |")
            
            self.report.append("")
    
    def _add_error_report(self):
        """Add error report section"""
        self.report.append("## 7. Error Report")
        self.report.append("")
        
        years = sorted(self.stats_by_year.keys())
        total_errors = sum(len(self.stats_by_year[y]['errors']) for y in years)
        
        # Summary table by year
        self.report.append("### Error Counts by Year")
        self.report.append("")
        self.report.append("| Year | Error Count |")
        self.report.append("|------|-------------|")
        for year in years:
            self.report.append(f"| {year} | {len(self.stats_by_year[year]['errors']):,} |")
        self.report.append("")
        
        if total_errors == 0:
            self.report.append("✅ No errors found during analysis.")
            self.report.append("")
            return
        
        # Detail by year
        for year in years:
            errors = self.stats_by_year[year]['errors']
            if errors:
                self.report.append(f"### {year} ({len(errors)} error(s))")
                self.report.append("")
                self.report.append("| CIK | Item | Error |")
                self.report.append("|-----|------|-------|")
                
                for error in errors[:10]:
                    error_msg = str(error['error'])[:50]
                    self.report.append(f"| {error['cik']} | {error['item']} | {error_msg} |")
                
                if len(errors) > 10:
                    self.report.append(f"| ... | ... | +{len(errors) - 10} more |")
                
                self.report.append("")
    
    def _add_conclusions(self):
        """Add conclusions section"""
        self.report.append("## 8. Key Insights")
        self.report.append("")
        
        years = sorted(self.stats_by_year.keys())
        total_filings = sum(sum(self.stats_by_year[y]['filings'].values()) 
                           for y in years)
        
        self.report.append(f"- **Total Filings Analyzed**: {total_filings:,}")
        self.report.append(f"- **Years Covered**: {years[0]}-{years[-1]}")
        
        # Find most extracted item
        all_items = {}
        for year in years:
            for item_num, item_data in self.stats_by_year[year]['items'].items():
                if item_num not in all_items:
                    all_items[item_num] = 0
                all_items[item_num] += item_data['success']
        
        best_item = max(all_items.items(), key=lambda x: x[1], default=('Unknown', 0))
        self.report.append(f"- **Most Extracted Item**: Item {best_item[0]} ({best_item[1]:,} total)")
        
        # Extra items note
        if self.extra_items:
            self.report.append(f"- **Extra Items Found**: {len(self.extra_items)} item types beyond ITEMS_10K")
        
        self.report.append("")
        self.report.append(f"*Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    def _write_report(self):
        """Write report to file"""
        output_dir = Path('stats')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'filing_analysis_{timestamp}.md'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.report))
        
        print(f"\n✅ Report generated: {output_file}")
        return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Analyze SEC filings and generate statistics report'
    )
    parser.add_argument(
        '--folder',
        default='sec_filings',
        help='Path to filings folder (default: sec_filings)'
    )
    
    args = parser.parse_args()
    
    try:
        analyzer = FilingAnalyzer(args.folder)
        report_file = analyzer.analyze()
        print(f"\nReport saved to: {report_file}")
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

    """Analyzes SEC filings and generates statistics report"""
    
    def __init__(self, filings_folder):
        self.filings_folder = Path(filings_folder)
        self.stats = {
            'filings': defaultdict(lambda: defaultdict(int)),
            'items': defaultdict(lambda: {
                'success': 0,
                'failed': 0,
                'headings': [],
                'bodies': [],
                'elements': [],
                'depths': []
            }),
            'files': {
                'sizes': [],
                'item_sizes': defaultdict(list),
                'structure_sizes': defaultdict(list)
            },
            'years': set(),
            'ciks': set()
        }
        self.report = []
        
    def analyze(self):
        """Run complete analysis"""
        print("Scanning filings folder...")
        self._scan_filings()
        
        print("Analyzing structure files...")
        self._analyze_structures()
        
        print("Generating report...")
        self._generate_report()
        
        return self._write_report()
    
    def _scan_filings(self):
        """Scan filings folder and collect statistics"""
        if not self.filings_folder.exists():
            raise Exception(f"Folder not found: {self.filings_folder}")
        
        # Walk through folder structure: {cik}/{year}/{filing_type}/
        for cik_folder in self.filings_folder.iterdir():
            if not cik_folder.is_dir() or cik_folder.name.startswith('.'):
                continue
            
            cik = cik_folder.name
            self.stats['ciks'].add(cik)
            
            for year_folder in cik_folder.iterdir():
                if not year_folder.is_dir():
                    continue
                
                try:
                    year = int(year_folder.name)
                except ValueError:
                    continue
                
                self.stats['years'].add(year)
                
                for filing_folder in year_folder.iterdir():
                    if not filing_folder.is_dir():
                        continue
                    
                    filing_type = filing_folder.name
                    self.stats['filings'][year][filing_type] += 1
                    
                    # Scan items folder
                    items_folder = filing_folder / 'items'
                    if items_folder.exists():
                        self._analyze_items_folder(items_folder, year, filing_type, cik)
                    
                    # Get filing size
                    for html_file in filing_folder.glob('*.[hH][tT][mM]*'):
                        if html_file.is_file():
                            self.stats['files']['sizes'].append(html_file.stat().st_size)
    
    def _analyze_items_folder(self, items_folder, year, filing_type, cik):
        """Analyze items in a filing folder"""
        for item_file in items_folder.glob('*_item*.json'):
            if item_file.name.endswith('_xtr.json'):
                continue  # Skip structure files for now
            
            # Extract item number from filename
            # Format: {CIK}_{year}_{filing_type}_item{num}.json
            parts = item_file.stem.split('_item')
            if len(parts) < 2:
                continue
            
            item_num = parts[-1]
            
            # Count by file existence instead of reading
            self.stats['items'][item_num]['success'] += 1
            self.stats['files']['item_sizes'][item_num].append(item_file.stat().st_size)
    
    def _analyze_structures(self):
        """Analyze structure files (*_xtr.json)"""
        for cik_folder in self.filings_folder.iterdir():
            if not cik_folder.is_dir() or cik_folder.name.startswith('.'):
                continue
            
            for year_folder in cik_folder.iterdir():
                if not year_folder.is_dir():
                    continue
                
                for filing_folder in year_folder.iterdir():
                    if not filing_folder.is_dir():
                        continue
                    
                    items_folder = filing_folder / 'items'
                    if not items_folder.exists():
                        continue
                    
                    for xtr_file in items_folder.glob('*_xtr.json'):
                        self._analyze_structure_file(xtr_file)
    
    def _analyze_structure_file(self, xtr_file):
        """Analyze a single structure file"""
        try:
            with open(xtr_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            item_num = data.get('item_number', 'Unknown')
            structure = data.get('structure', [])
            
            if structure:
                # Count headings and bodies
                headings = self._count_headings(structure)
                bodies = self._count_bodies(structure)
                depth = self._max_depth(structure)
                
                self.stats['items'][item_num]['headings'].append(headings)
                self.stats['items'][item_num]['bodies'].append(bodies)
                self.stats['items'][item_num]['elements'].append(len(self._flatten_structure(structure)))
                self.stats['items'][item_num]['depths'].append(depth)
                
                self.stats['files']['structure_sizes'][item_num].append(xtr_file.stat().st_size)
        except Exception as e:
            pass
    
    def _count_headings(self, structure, bold_only=False):
        """Count headings in structure"""
        count = 0
        for elem in structure:
            if elem.get('type') in ['heading', 'bold_heading']:
                if not bold_only or elem.get('type') == 'bold_heading':
                    count += 1
            if 'children' in elem:
                count += self._count_headings(elem['children'], bold_only)
        return count
    
    def _count_bodies(self, structure):
        """Count non-empty bodies in structure"""
        count = 0
        for elem in structure:
            if elem.get('body', '').strip():
                count += 1
            if 'children' in elem:
                count += self._count_bodies(elem['children'])
        return count
    
    def _max_depth(self, structure, depth=0):
        """Get maximum nesting depth"""
        max_d = depth
        for elem in structure:
            if 'children' in elem:
                max_d = max(max_d, self._max_depth(elem['children'], depth + 1))
        return max_d
    
    def _flatten_structure(self, structure):
        """Flatten structure to count all elements"""
        elements = []
        for elem in structure:
            elements.append(elem)
            if 'children' in elem:
                elements.extend(self._flatten_structure(elem['children']))
        return elements
    
    def _generate_report(self):
        """Generate markdown report"""
        self.report = []
        
        # Header
        self.report.append("# Filing Analysis Report")
        self.report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.report.append("")
        
        # Executive Summary
        self._add_executive_summary()
        
        # Year-by-Year Overview
        self._add_year_overview()
        
        # Item Extraction Analysis
        self._add_item_extraction()
        
        # File Statistics
        self._add_file_statistics()
        
        # Structure Extraction Analysis
        self._add_structure_analysis()
        
        # Headings & Bodies Analysis
        self._add_headings_bodies_analysis()
        
        # Item Statistics Heatmap
        self._add_item_heatmap()
        
        # Conclusions
        self._add_conclusions()
    
    def _add_executive_summary(self):
        """Add executive summary section"""
        self.report.append("## 1. Executive Summary")
        self.report.append("")
        
        total_filings = sum(sum(v.values()) for v in self.stats['filings'].values())
        years = sorted(self.stats['years'])
        total_size = sum(self.stats['files']['sizes']) / (1024**3)  # GB
        
        self.report.append(f"- **Total Filings Analyzed**: {total_filings:,}")
        self.report.append(f"- **Unique Companies (CIKs)**: {len(self.stats['ciks']):,}")
        self.report.append(f"- **Years Covered**: {years[0]}-{years[-1]}")
        self.report.append(f"- **Total Storage Used**: {total_size:.1f} GB")
        self.report.append(f"- **Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.report.append("")
    
    def _add_year_overview(self):
        """Add year-by-year overview"""
        self.report.append("## 2. Year-by-Year Overview")
        self.report.append("")
        
        for year in sorted(self.stats['years']):
            total = sum(self.stats['filings'][year].values())
            self.report.append(f"### {year} ({total:,} filings)")
            self.report.append("")
            
            for filing_type, count in self.stats['filings'][year].items():
                self.report.append(f"- **{filing_type}**: {count:,} filings")
            
            self.report.append("")
    
    def _add_item_extraction(self):
        """Add item extraction analysis"""
        self.report.append("## 3. Item Extraction Analysis")
        self.report.append("")
        
        self.report.append("### Extraction Success Rates")
        self.report.append("")
        self.report.append("| Item | Success | Failed | Success Rate |")
        self.report.append("|------|---------|--------|--------------|")
        
        for item in sorted(self.stats['items'].keys()):
            success = self.stats['items'][item]['success']
            failed = self.stats['items'][item]['failed']
            total = success + failed
            if total > 0:
                rate = (success / total) * 100
                self.report.append(f"| Item {item} | {success:,} | {failed:,} | {rate:.1f}% |")
        
        self.report.append("")
    
    def _add_file_statistics(self):
        """Add file statistics"""
        self.report.append("## 4. File Statistics")
        self.report.append("")
        
        if self.stats['files']['sizes']:
            sizes_mb = [s / (1024**2) for s in self.stats['files']['sizes']]
            avg_size = statistics.mean(sizes_mb)
            median_size = statistics.median(sizes_mb)
            min_size = min(sizes_mb)
            max_size = max(sizes_mb)
            
            self.report.append("### Filing Size Metrics")
            self.report.append("")
            self.report.append(f"- **Average Filing Size**: {avg_size:.2f} MB")
            self.report.append(f"- **Median Filing Size**: {median_size:.2f} MB")
            self.report.append(f"- **Min Filing Size**: {min_size:.2f} MB")
            self.report.append(f"- **Max Filing Size**: {max_size:.2f} MB")
            self.report.append("")
        
        self.report.append("### Item JSON File Sizes")
        self.report.append("")
        self.report.append("| Item | Avg Size | Min Size | Max Size |")
        self.report.append("|------|----------|----------|----------|")
        
        for item in sorted(self.stats['files']['item_sizes'].keys()):
            sizes = [s / 1024 for s in self.stats['files']['item_sizes'][item]]
            if sizes:
                avg = statistics.mean(sizes)
                min_s = min(sizes)
                max_s = max(sizes)
                self.report.append(f"| Item {item} | {avg:.1f} KB | {min_s:.1f} KB | {max_s:.1f} KB |")
        
        self.report.append("")
    
    def _add_structure_analysis(self):
        """Add structure extraction analysis"""
        self.report.append("## 5. Structure Extraction Analysis")
        self.report.append("")
        
        self.report.append("### Structure Files Created")
        self.report.append("")
        
        total_structures = sum(len(v) for v in self.stats['files']['structure_sizes'].values())
        self.report.append(f"- **Total Structure Files**: {total_structures:,}")
        self.report.append("")
        
        self.report.append("### Average Metrics by Item")
        self.report.append("")
        self.report.append("| Item | Avg Depth | Max Depth | Avg Elements |")
        self.report.append("|------|-----------|-----------|--------------|")
        
        for item in sorted(self.stats['items'].keys()):
            depths = self.stats['items'][item]['depths']
            elements = self.stats['items'][item]['elements']
            
            if depths and elements:
                avg_depth = statistics.mean(depths)
                max_depth = max(depths) if depths else 0
                avg_elems = statistics.mean(elements)
                self.report.append(f"| Item {item} | {avg_depth:.1f} | {max_depth} | {avg_elems:.0f} |")
        
        self.report.append("")
    
    def _add_headings_bodies_analysis(self):
        """Add headings and bodies analysis"""
        self.report.append("## 6. Headings & Bodies Analysis by Item")
        self.report.append("")
        
        self.report.append("### Item-by-Item Heading & Body Breakdown")
        self.report.append("")
        self.report.append("| Item | Avg Headings | Avg Bodies | H/B Ratio | Avg Elements |")
        self.report.append("|------|-------------|-----------|-----------|--------------|")
        
        for item in sorted(self.stats['items'].keys()):
            headings = self.stats['items'][item]['headings']
            bodies = self.stats['items'][item]['bodies']
            elements = self.stats['items'][item]['elements']
            
            if headings and bodies:
                avg_h = statistics.mean(headings)
                avg_b = statistics.mean(bodies)
                ratio = avg_h / avg_b if avg_b > 0 else 0
                avg_e = statistics.mean(elements)
                
                self.report.append(f"| Item {item} | {avg_h:.1f} | {avg_b:.1f} | 1:{ratio:.1f} | {avg_e:.0f} |")
        
        self.report.append("")
        
        # Global metrics
        all_headings = []
        all_bodies = []
        all_elements = []
        
        for item in self.stats['items'].values():
            all_headings.extend(item['headings'])
            all_bodies.extend(item['bodies'])
            all_elements.extend(item['elements'])
        
        self.report.append("### Global Metrics")
        self.report.append("")
        
        if all_headings:
            self.report.append(f"- **Total Headings Extracted**: {sum(all_headings):,}")
            self.report.append(f"- **Avg Headings per Filing**: {statistics.mean(all_headings):.0f}")
        
        if all_bodies:
            self.report.append(f"- **Total Bodies Extracted**: {sum(all_bodies):,}")
            self.report.append(f"- **Avg Bodies per Filing**: {statistics.mean(all_bodies):.0f}")
        
        if all_headings and all_bodies:
            ratio = sum(all_headings) / sum(all_bodies) if sum(all_bodies) > 0 else 0
            self.report.append(f"- **Overall Heading/Body Ratio**: 1:{ratio:.2f}")
        
        if all_elements:
            self.report.append(f"- **Total Elements**: {sum(all_elements):,}")
            self.report.append(f"- **Avg Elements per Filing**: {statistics.mean(all_elements):.0f}")
        
        self.report.append("")
    
    def _add_item_heatmap(self):
        """Add item statistics heatmap"""
        self.report.append("## 7. Item Statistics Summary")
        self.report.append("")
        
        self.report.append("### Structure Complexity by Item Type")
        self.report.append("")
        self.report.append("| Item | Count | Avg Depth | Avg Headings | Avg Bodies | Avg Elements |")
        self.report.append("|------|-------|-----------|-------------|-----------|--------------|")
        
        for item in sorted(self.stats['items'].keys()):
            item_data = self.stats['items'][item]
            count = item_data['success']
            
            if count > 0:
                avg_depth = statistics.mean(item_data['depths']) if item_data['depths'] else 0
                avg_h = statistics.mean(item_data['headings']) if item_data['headings'] else 0
                avg_b = statistics.mean(item_data['bodies']) if item_data['bodies'] else 0
                avg_e = statistics.mean(item_data['elements']) if item_data['elements'] else 0
                
                self.report.append(f"| Item {item} | {count:,} | {avg_depth:.1f} | {avg_h:.0f} | {avg_b:.0f} | {avg_e:.0f} |")
        
        self.report.append("")
    
    def _add_conclusions(self):
        """Add conclusions section"""
        self.report.append("## 8. Key Insights")
        self.report.append("")
        
        total_filings = sum(sum(v.values()) for v in self.stats['filings'].values())
        
        self.report.append(f"- **Total Filings Analyzed**: {total_filings:,} from {len(self.stats['ciks']):,} companies")
        self.report.append(f"- **Years Covered**: {min(self.stats['years'])}-{max(self.stats['years'])}")
        
        # Find most extracted item
        best_item = max(self.stats['items'].items(), 
                       key=lambda x: x[1]['success'], 
                       default=('Unknown', {'success': 0}))
        
        self.report.append(f"- **Most Extracted Item**: Item {best_item[0]} ({best_item[1]['success']:,} filings)")
        
        # Find deepest structures
        deepest_item = max(self.stats['items'].items(),
                          key=lambda x: max(x[1]['depths']) if x[1]['depths'] else 0,
                          default=('Unknown', {'depths': [0]}))
        
        max_d = max(deepest_item[1]['depths']) if deepest_item[1]['depths'] else 0
        self.report.append(f"- **Most Complex Structure**: Item {deepest_item[0]} (max depth: {max_d})")
        
        self.report.append("")
        self.report.append(f"*Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    def _write_report(self):
        """Write report to file"""
        output_dir = Path('stats')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'filing_analysis_{timestamp}.md'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.report))
        
        print(f"\n✅ Report generated: {output_file}")
        return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Analyze SEC filings and generate statistics report'
    )
    parser.add_argument(
        '--folder',
        default='sec_filings',
        help='Path to filings folder (default: sec_filings)'
    )
    
    args = parser.parse_args()
    
    try:
        analyzer = FilingAnalyzer(args.folder)
        report_file = analyzer.analyze()
        print(f"\nReport saved to: {report_file}")
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
