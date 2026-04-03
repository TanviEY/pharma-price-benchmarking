# backend.py — consolidated backend logic (flat functions, no classes)

import json
import os
import glob
import fnmatch
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from thefuzz import fuzz
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

# ── Settings ──────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent
_CONFIG_DIR = _PROJECT_ROOT / "config"


def _load_molecule_mapping() -> Dict:
    config_path = _CONFIG_DIR / "molecule_mapping.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


MOLECULE_MAPPING = _load_molecule_mapping()

# ── Data Loading ───────────────────────────────────────────────────────────────


def load_cipla_grn(file_path: str, api_filter: Optional[str] = None) -> pd.DataFrame:
    """Load Cipla GRN data and filter by API"""
    try:
        df = pd.read_excel(file_path)
        if api_filter and 'api_family' in df.columns:
            df = df[df['api_family'].str.contains(api_filter, case=False, na=False)]
        return df
    except Exception as e:
        raise ValueError(f"Error loading Cipla file {file_path}: {str(e)}")


def load_multiple_files(file_list: list) -> pd.DataFrame:
    """Load and concatenate multiple Excel files"""
    dfs = []
    for file_path in file_list:
        try:
            df = pd.read_excel(file_path)
            df['source_file'] = file_path
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {str(e)}")

    if not dfs:
        raise ValueError("No files could be loaded")

    return pd.concat(dfs, ignore_index=True)

# ── Data Processing ────────────────────────────────────────────────────────────


def extract_grade_spec(item_text: str) -> str:
    """
    Extract grade/spec from ITEM description.
    Hierarchy (highest wins): USP > EP > IP. Defaults to USP if none found.
    """
    if not isinstance(item_text, str):
        return 'USP'
    item_lower = item_text.lower()
    if 'usp' in item_lower:
        return 'USP'
    if 'ep' in item_lower:
        return 'EP'
    if 'ip' in item_lower:
        return 'IP'
    return 'USP'


def extract_yyyymm(date_col) -> str:
    """Extract yyyymm from date"""
    try:
        if isinstance(date_col, (int, float)):
            date_obj = pd.to_datetime(str(int(date_col)), format='%Y%m%d')
        elif isinstance(date_col, str):
            date_obj = pd.to_datetime(date_col)
        else:
            date_obj = date_col
        return date_obj.strftime('%Y%m')
    except Exception:
        return None


def apply_outlier_filters(df: pd.DataFrame, cipla_baseline: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Apply outlier filters using EXIM avg qty and Cipla price baseline
    1. QTY >= 10% of EXIM Avg Qty
    2. unit_price within Cipla Price ± 30%
    Returns: (filtered_df, outlier_df, stats_dict)
    """
    original_df = df.copy()
    original_count = len(df)

    # Filter 1: Quantity threshold
    exim_avg_qty = df['QTY'].mean()
    min_qty = exim_avg_qty * 0.10
    df = df[df['QTY'] >= min_qty]

    # Filter 2: Price variance
    cipla_price = cipla_baseline['avg_price']
    price_lower = cipla_price * 0.70  # -30%
    price_upper = cipla_price * 1.30  # +30%

    # Calculate per-unit price
    df['unit_price'] = df['TOTAL_VALUE'] / df['QTY']
    df = df[(df['unit_price'] >= price_lower) & (df['unit_price'] <= price_upper)]

    filtered_count = len(df)
    removed = original_count - filtered_count

    # Build outlier df: rows in original that are NOT in filtered
    outlier_df = original_df[~original_df.index.isin(df.index)].copy()
    # Add reason columns for validation
    outlier_df['unit_price'] = outlier_df['TOTAL_VALUE'] / outlier_df['QTY']
    outlier_df['outlier_reason_qty'] = outlier_df['QTY'] < min_qty
    outlier_df['outlier_reason_price'] = ~((outlier_df['unit_price'] >= price_lower) & (outlier_df['unit_price'] <= price_upper))

    return df, outlier_df, {
        'original_count': original_count,
        'filtered_count': filtered_count,
        'removed_count': removed,
        'removal_percentage': (removed / original_count * 100) if original_count > 0 else 0
    }


def prepare_molecule_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare molecule EXIM data - works for any molecule file"""
    df = df.copy()

    # Find date column
    date_col = None
    for col in ['BE_DATE', 'be_date', 'DATE', 'date', 'BILL_DATE']:
        if col in df.columns:
            date_col = col
            break
    if date_col is None:
        raise ValueError(f"No date column found. Available columns: {list(df.columns)}")
    df['yyyymm'] = df[date_col].apply(extract_yyyymm)

    # Find item/description column for grade spec
    item_col = None
    for col in ['ITEM', 'item', 'ITEM_DESC', 'DESCRIPTION', 'PRODUCT']:
        if col in df.columns:
            item_col = col
            break
    if item_col:
        df['GRADE_SPEC'] = df[item_col].apply(extract_grade_spec)
    else:
        df['GRADE_SPEC'] = 'USP'

    # Ensure QTY is numeric
    df['QTY'] = pd.to_numeric(df['QTY'], errors='coerce')
    df['TOTAL_VALUE'] = pd.to_numeric(df['TOTAL_VALUE'], errors='coerce')

    # Remove rows with missing critical values
    df = df.dropna(subset=['yyyymm', 'QTY', 'TOTAL_VALUE'])

    return df


def prepare_cipla_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare Cipla GRN data"""
    df = df.copy()

    # Extract date
    df['yyyymm'] = df['posting_date_in_the_document'].apply(extract_yyyymm)

    # Ensure numeric columns
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['actual_spend_inr'] = pd.to_numeric(df['actual_spend_inr'], errors='coerce')

    # Remove rows with missing critical values
    df = df.dropna(subset=['yyyymm', 'quantity', 'actual_spend_inr'])

    return df


def calculate_cipla_baseline(cipla_df: pd.DataFrame) -> Dict:
    """Calculate baseline metrics from Cipla data"""
    return {
        'avg_qty': cipla_df['quantity'].mean(),
        'avg_price': (cipla_df['actual_spend_inr'] / cipla_df['quantity']).mean(),
        'min_qty': cipla_df['quantity'].min(),
        'max_qty': cipla_df['quantity'].max(),
        'total_records': len(cipla_df)
    }


def _aggregate_entity(df: pd.DataFrame, entity_col: str, entity_alias: str, source: str) -> pd.DataFrame:
    """Shared aggregation logic for supplier and buyer views."""
    agg_df = df.groupby([entity_col, 'yyyymm', 'UQC', 'GRADE_SPEC']).agg(
        Sum_of_QTY=('QTY', 'sum'),
        Sum_of_TOTAL_VALUE=('TOTAL_VALUE', 'sum')
    ).reset_index()
    agg_df.rename(columns={entity_col: entity_alias, 'UQC': 'uom'}, inplace=True)
    agg_df['Avg_PRICE'] = agg_df['Sum_of_TOTAL_VALUE'] / agg_df['Sum_of_QTY']
    agg_df['source'] = source
    return agg_df[[entity_alias, 'yyyymm', 'uom', 'GRADE_SPEC', 'Sum_of_QTY', 'Sum_of_TOTAL_VALUE', 'Avg_PRICE', 'source']]


def aggregate_supplier(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate data by Supplier"""
    return _aggregate_entity(df, 'Supp_Name', 'supplier', 'Supplier')


def aggregate_buyer(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate data by Buyer (Importer)"""
    return _aggregate_entity(df, 'IMPORTER', 'buyer', 'Buyer')


def aggregate_cipla(cipla_df: pd.DataFrame, molecule_name: str = 'unknown') -> pd.DataFrame:
    """Aggregate Cipla data"""
    agg_df = cipla_df.groupby(['yyyymm', 'base_unit_of_measure', 'grade_spec']).agg({
        'quantity': 'sum',
        'actual_spend_inr': 'sum'
    }).reset_index()

    agg_df.rename(columns={
        'base_unit_of_measure': 'uom',
        'grade_spec': 'GRADE_SPEC',
        'quantity': 'Sum_of_QTY',
        'actual_spend_inr': 'Sum_of_TOTAL_VALUE'
    }, inplace=True)

    agg_df['Avg_PRICE'] = agg_df['Sum_of_TOTAL_VALUE'] / agg_df['Sum_of_QTY']
    agg_df['source'] = 'Cipla'
    agg_df['api'] = molecule_name.upper()

    return agg_df[['api', 'yyyymm', 'uom', 'GRADE_SPEC', 'Sum_of_QTY', 'Sum_of_TOTAL_VALUE', 'Avg_PRICE', 'source']]

# ── File Discovery ─────────────────────────────────────────────────────────────


def discover_molecule_files(molecule: str, data_dir: str, molecule_mapping: Dict) -> List[str]:
    """
    Discover all files for a given molecule

    Args:
        molecule: Molecule name (e.g., 'azithromycin')
        data_dir: Path to the data directory
        molecule_mapping: Molecule mapping config dict

    Returns:
        List of file paths matching the molecule patterns
    """
    if molecule not in molecule_mapping['molecules']:
        return []

    patterns = molecule_mapping['molecules'][molecule]['file_patterns']
    found_files = set()
    data_path = Path(data_dir)

    # Case-sensitive glob first
    for pattern in patterns:
        file_path = os.path.join(data_path, pattern)
        found_files.update(glob.glob(file_path))

    # Case-insensitive fallback: scan directory
    if data_path.exists():
        for f in data_path.iterdir():
            if f.is_file():
                for pattern in patterns:
                    if fnmatch.fnmatch(f.name.lower(), pattern.lower()):
                        found_files.add(str(f))

    return sorted(found_files)


def discover_cipla_file(data_dir: str) -> Optional[str]:
    """Discover Cipla GRN file"""
    cipla_patterns = ["cipla_api_grn*.xlsx", "cipla_grn*.xlsx"]
    data_path = Path(data_dir)

    for pattern in cipla_patterns:
        file_path = os.path.join(data_path, pattern)
        files = glob.glob(file_path)
        if files:
            return files[0]

    return None


def get_available_molecules(data_dir: str, molecule_mapping: Dict) -> Dict[str, Dict]:
    """Get all available molecules with their file count"""
    available = {}
    data_path = Path(data_dir)

    for mol_name, mol_config in molecule_mapping['molecules'].items():
        patterns = mol_config.get("file_patterns", [f"*{mol_name}*"])
        found_files = set()

        # Case-sensitive glob first
        for pattern in patterns:
            file_path = os.path.join(data_path, pattern)
            found_files.update(glob.glob(file_path))

        # Case-insensitive fallback: scan directory
        if data_path.exists():
            for f in data_path.iterdir():
                if f.is_file():
                    for pattern in patterns:
                        if fnmatch.fnmatch(f.name.lower(), pattern.lower()):
                            found_files.add(str(f))

        if found_files:
            available[mol_name] = {
                "description": mol_config.get("description", ""),
                "file_count": len(found_files),
                "cipla_available": "cipla_api_filter" in mol_config
            }

    return available


def get_molecule_file_info(molecule: str, data_dir: str, molecule_mapping: Dict) -> Dict:
    """Get detailed file information for a molecule"""
    files = discover_molecule_files(molecule, data_dir, molecule_mapping)

    info = {
        'molecule': molecule,
        'total_files': len(files),
        'files': [],
        'cipla_available': discover_cipla_file(data_dir) is not None,
        'size_bytes': 0,
        'last_modified': None
    }

    for file in files:
        file_stat = os.stat(file)
        info['files'].append({
            'name': os.path.basename(file),
            'path': file,
            'size_bytes': file_stat.st_size,
            'modified': file_stat.st_mtime
        })
        info['size_bytes'] += file_stat.st_size

    return info

# ── Fuzzy Matching ─────────────────────────────────────────────────────────────


def _score_query(query: str, candidate: str) -> int:
    """Score a query against a single candidate string."""
    q, c = query.lower(), candidate.lower()
    if q == c:
        return 100
    score = max(fuzz.ratio(q, c), fuzz.partial_ratio(q, c))
    # Substring bonus
    if q in c or c in q:
        score = max(score, 60)
    return score


def match_molecule_input(user_input: str, molecule_mapping: Dict, threshold: int = 70) -> List[Tuple[str, int]]:
    """
    Find matching molecules based on user input.
    Returns a list of (molecule_name, confidence_score) tuples sorted by score.
    """
    if not user_input.strip():
        return []

    query = user_input.lower().strip()
    best_scores: Dict[str, int] = {}

    for mol_name, mol_data in molecule_mapping["molecules"].items():
        candidates = [mol_name] + mol_data.get("aliases", [])
        top_score = max(_score_query(query, c) for c in candidates)
        if top_score >= threshold:
            best_scores[mol_name] = max(best_scores.get(mol_name, 0), top_score)

    return sorted(best_scores.items(), key=lambda x: x[1], reverse=True)


def get_suggestions(user_input: str, molecule_mapping: Dict, top_n: int = 5, threshold: int = 40) -> List[Tuple[str, int]]:
    """
    Return up to top_n molecule suggestions for the given input.
    Uses a lower threshold (40) and partial matching to be more forgiving.
    If no suggestions found, returns top_n molecules alphabetically (score=0).
    """
    if not user_input.strip():
        top_mols = sorted(molecule_mapping.get("molecules", {}).keys())[:top_n]
        return [(m, 0) for m in top_mols]

    query = user_input.lower().strip()
    best_scores: Dict[str, int] = {}

    for mol_name, mol_data in molecule_mapping["molecules"].items():
        candidates = [mol_name] + mol_data.get("aliases", [])
        top_score = max(_score_query(query, c) for c in candidates)
        if top_score >= threshold:
            best_scores[mol_name] = max(best_scores.get(mol_name, 0), top_score)

    if not best_scores:
        # No suggestions found — return top_n alphabetically
        top_mols = sorted(molecule_mapping.get("molecules", {}).keys())[:top_n]
        return [(m, 0) for m in top_mols]

    sorted_suggestions = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_suggestions[:top_n]


def get_top_match(user_input: str, molecule_mapping: Dict, threshold: int = 70) -> Optional[str]:
    """Get the single best match above the configured threshold."""
    matches = match_molecule_input(user_input, molecule_mapping, threshold)
    return matches[0][0] if matches else None


def get_aliases(molecule: str, molecule_mapping: Dict) -> List[str]:
    """Get all aliases for a molecule"""
    if molecule in molecule_mapping.get("molecules", {}):
        return molecule_mapping["molecules"][molecule].get("aliases", [])
    return []

# ── Pipeline ───────────────────────────────────────────────────────────────────


def run_processing_pipeline(molecule_name: str, data_dir: str) -> Dict:
    """
    Universal pipeline for any molecule
    """
    try:
        # Step 1: Discover files
        mol_files = discover_molecule_files(molecule_name, data_dir, MOLECULE_MAPPING)
        cipla_file = discover_cipla_file(data_dir)

        if not mol_files:
            return {
                'status': 'failed',
                'errors': [f'No data files found for molecule: {molecule_name}']
            }
        if not cipla_file:
            return {
                'status': 'failed',
                'errors': ['Cipla GRN file not found']
            }

        # Step 2: Load data
        molecule_df = load_multiple_files(mol_files)
        api_filter = MOLECULE_MAPPING['molecules'][molecule_name]['cipla_api_filter']
        cipla_df = load_cipla_grn(cipla_file, api_filter)

        raw_record_count = len(molecule_df)

        # Step 3: Prepare data
        molecule_df = prepare_molecule_data(molecule_df)
        cipla_df = prepare_cipla_data(cipla_df)

        # Step 4: Calculate baselines
        cipla_baseline = calculate_cipla_baseline(cipla_df)

        # Step 5: Filter outliers (returns tuple of (filtered_df, outlier_df, stats))
        molecule_df_filtered, outlier_df, filter_stats = apply_outlier_filters(molecule_df, cipla_baseline)

        # Step 6: Aggregate
        supplier_agg = aggregate_supplier(molecule_df_filtered)
        buyer_agg = aggregate_buyer(molecule_df_filtered)
        cipla_agg = aggregate_cipla(cipla_df, molecule_name)

        # Step 7: Build consolidated DataFrame
        shared_cols = ['entity_name', 'yyyymm', 'uom', 'GRADE_SPEC',
                       'Sum_of_QTY', 'Sum_of_TOTAL_VALUE', 'Avg_PRICE', 'source']
        supplier_agg['entity_name'] = supplier_agg['supplier']
        buyer_agg['entity_name'] = buyer_agg['buyer']
        cipla_agg['entity_name'] = cipla_agg['api']
        consolidated = pd.concat(
            [supplier_agg[shared_cols], buyer_agg[shared_cols], cipla_agg[shared_cols]],
            ignore_index=True
        )

        # Step 8: Save outputs
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        supplier_agg.to_csv(processed_dir / f"{molecule_name}_supplier.csv", index=False)
        buyer_agg.to_csv(processed_dir / f"{molecule_name}_buyer.csv", index=False)
        cipla_agg.to_csv(processed_dir / f"cipla_{molecule_name}.csv", index=False)
        outlier_df.to_csv(processed_dir / f"outlier_{molecule_name}.csv", index=False)

        return {
            'status': 'success',
            'errors': [],
            'metadata': {
                'files_loaded': mol_files,
                'raw_record_count': raw_record_count,
                'filter_stats': filter_stats,
                'cipla_baseline': cipla_baseline
            },
            'data': {
                'supplier': supplier_agg,
                'buyer': buyer_agg,
                'cipla': cipla_agg,
                'consolidated': consolidated,
                'outlier': outlier_df,
            }
        }

    except Exception as e:
        return {
            'status': 'failed',
            'errors': [str(e)]
        }

# ── Utils ──────────────────────────────────────────────────────────────────────


def format_currency(value: float) -> str:
    """Format value as currency"""
    return f"₹{value:,.2f}"


def format_percentage(value: float) -> str:
    """Format as percentage"""
    return f"{value:.2f}%"


def calculate_price_variance(actual_price: float, baseline_price: float) -> float:
    """Calculate price variance from baseline"""
    if baseline_price == 0:
        return 0
    return ((actual_price - baseline_price) / baseline_price) * 100


def get_grade_spec_options(df: pd.DataFrame) -> List[str]:
    """Get unique grade/spec values"""
    return sorted(df['GRADE_SPEC'].unique().tolist())


def get_uom_options(df: pd.DataFrame) -> List[str]:
    """Get unique UOM values"""
    return sorted(df['uom'].unique().tolist())


def get_date_range(df: pd.DataFrame) -> tuple:
    """Get min and max dates"""
    dates = pd.to_datetime(df['yyyymm'], format='%Y%m')
    return dates.min(), dates.max()


def filter_dataframe(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """Apply filters to dataframe"""
    filtered = df.copy()

    if 'date_range' in filters and filters['date_range']:
        min_date, max_date = filters['date_range']
        filtered = filtered[
            (filtered['yyyymm'] >= min_date.strftime('%Y%m')) &
            (filtered['yyyymm'] <= max_date.strftime('%Y%m'))
        ]

    if 'grade_specs' in filters and filters['grade_specs']:
        filtered = filtered[filtered['GRADE_SPEC'].isin(filters['grade_specs'])]

    if 'uoms' in filters and filters['uoms']:
        filtered = filtered[filtered['uom'].isin(filters['uoms'])]

    if 'sources' in filters and filters['sources']:
        filtered = filtered[filtered['source'].isin(filters['sources'])]

    return filtered
