# backend.py — consolidated backend logic (flat functions, no classes)

import json
import os
import glob
import fnmatch
import re
import warnings
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from thefuzz import fuzz
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')


def _load_local_env_file() -> None:
    """Load key=value pairs from a .env file in the project root into os.environ."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value
    except Exception:
        pass


_load_local_env_file()


# ── LLM provider: OpenAI (SSL-disabled for corporate/Zscaler networks) ────────
#  Add OPENAI_API_KEY to your .env to enable AI features.


class NoSSLOpenAIClient:
    """OpenAI client with SSL verification disabled for corporate proxy environments."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def chat_completion(self, messages, model="gpt-4o-mini", max_tokens=1000, temperature=0.1):
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=self.headers,
                json=payload,
                verify=False,
                timeout=60,
            )
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                raise Exception("Invalid API key. Please check your environment variables.")
            elif response.status_code == 429:
                raise Exception("Rate limit exceeded or insufficient credits in OpenAI account.")
            else:
                raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as exc:
            raise Exception(f"Request failed: {exc}")


_OPENAI_CLIENT = None
_OPENAI_AVAILABLE = False
try:
    _OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if _OPENAI_API_KEY:
        _OPENAI_CLIENT = NoSSLOpenAIClient(_OPENAI_API_KEY)
        _OPENAI_AVAILABLE = True
except Exception:
    pass

_LLM_AVAILABLE = _OPENAI_AVAILABLE
_LLM_PROVIDER = "openai" if _OPENAI_AVAILABLE else "none"
_LLM_INIT_ERROR = (
    "" if _LLM_AVAILABLE
    else "No LLM available. Set OPENAI_API_KEY in your .env file."
)


def _llm_generate(prompt: str) -> str:
    """
    Send a prompt to OpenAI and return the text response.
    Returns '' if the client is unavailable or the call fails.
    """
    if _OPENAI_AVAILABLE and _OPENAI_CLIENT is not None:
        try:
            result = _OPENAI_CLIENT.chat_completion(
                messages=[{"role": "user", "content": prompt}]
            )
            return result["choices"][0]["message"]["content"].strip()
        except Exception:
            pass

    return ""


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

_UOM_CANONICAL_MAP = {
    "KGS": "KG",
    "GMS": "GM",
    "GRM": "GM",
    "GRMS": "GM",
    "LTR": "LT",
    "LTRS": "LT",
    "MLS": "ML",
    "MGS": "MG",
}

_KNOWN_UOMS = set(_UOM_CANONICAL_MAP.values()) | set(_UOM_CANONICAL_MAP.keys())


def normalize_uom(uom: str) -> str:
    """Normalize UOM string to a canonical uppercase form."""
    if not isinstance(uom, str):
        return uom
    uom = uom.strip().upper()
    return _UOM_CANONICAL_MAP.get(uom, uom)


def llm_normalize_uom(df: pd.DataFrame, uom_col: str) -> pd.DataFrame:
    """
    Normalize UOM column using Gemini to map non-standard values to canonical forms.
    Falls back to normalize_uom() if Gemini is unavailable.
    """
    if uom_col not in df.columns:
        return df

    # Always apply rule-based normalization first
    df[uom_col] = df[uom_col].apply(normalize_uom)

    if not _LLM_AVAILABLE:
        return df

    non_standard_mask = ~df[uom_col].apply(
        lambda v: (isinstance(v, str) and v in _KNOWN_UOMS)
    )
    non_standard_values = df.loc[non_standard_mask, uom_col].dropna().unique()

    if len(non_standard_values) == 0:
        return df

    try:
        values_list = "\n".join(f"- {v}" for v in non_standard_values)
        prompt = (
            "You are a pharmaceutical data normalizer. "
            "Map each of the following non-standard Unit of Measure (UOM) values to a canonical short uppercase form "
            "(e.g. KG, GM, ML, LT, MG, NOS, TAB, CAP, AMP, VIAL, PKT). "
            "Return a JSON object mapping each original value to its normalized UOM. "
            "Example: {\"kilogram\": \"KG\", \"grams\": \"GM\"}.\n"
            f"Values to normalize:\n{values_list}"
        )
        text = _llm_generate(prompt)
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            mapping = json.loads(json_match.group())
            clean_mapping = {
                orig: normalized.strip().upper()
                for orig, normalized in mapping.items()
                if isinstance(normalized, str) and normalized.strip()
            }
            if clean_mapping:
                df[uom_col] = df[uom_col].replace(clean_mapping)
    except Exception:
        pass  # fallback: rule-based normalization already applied above

    return df


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


def llm_extract_grade_spec(item_text: str) -> str:
    """
    Use Gemini to extract grade/spec from ITEM description.
    Falls back to extract_grade_spec() if Gemini is unavailable.
    Valid grades: USP, EP, IP, IH, BP
    """
    if not _LLM_AVAILABLE or not isinstance(item_text, str):
        return extract_grade_spec(item_text)
    try:
        prompt = (
            "You are a pharmaceutical data parser. "
            "Extract the pharmacopoeia grade from this API item description. "
            "Return ONLY one of: USP, EP, IP, IH, BP. "
            "If the description mentions 'inhouse' or 'IH' or 'no set pharmacopoeia', return IH. "
            "If unclear, return USP.\n"
            f"Item: {item_text}"
        )
        grade = _llm_generate(prompt).upper()
        if grade in ("USP", "EP", "IP", "IH", "BP"):
            return grade
        return extract_grade_spec(item_text)
    except Exception:
        return extract_grade_spec(item_text)


# Cache for LLM item relevance results keyed by (molecule, item_value)
_item_relevance_cache: Dict[Tuple[str, str], dict] = {}


def llm_check_item_relevance(molecule: str, item_value: str) -> dict:
    """
    Use Gemini to determine if item_value is a direct pharmaceutical form of molecule.
    Returns a dict: {"is_relevant": bool, "outlier_flag": bool, "outlier_reason": str}
    The LLM prompt is the sole decision-maker; no hardcoded keyword rules are used.
    If Gemini is unavailable or the response cannot be parsed, the item is treated as
    relevant (safe pass-through) to avoid false-positive outlier flags.
    Results are cached per (molecule, item_value) pair to avoid redundant LLM calls.
    """
    _relevant = {"is_relevant": True, "outlier_flag": False, "outlier_reason": ""}

    if not isinstance(item_value, str) or not item_value.strip():
        return _relevant

    cache_key = (molecule.lower().strip(), item_value.strip())
    if cache_key in _item_relevance_cache:
        return _item_relevance_cache[cache_key]

    if not _LLM_AVAILABLE:
        _item_relevance_cache[cache_key] = _relevant
        return _relevant

    try:
        prompt = (
            f'You are a pharmaceutical expert analyzing import/export trade data.\n'
            f'Molecule of interest: "{molecule}"\n'
            f'Item description in the dataset: "{item_value}"\n\n'
            f'Determine if this item directly represents "{molecule}" as an active pharmaceutical '
            f'ingredient (API), its salt, hydrate, ester, polymorph, co-crystal, or finished dosage form.\n\n'
            f'Flag as NOT RELEVANT (outlier) if ANY of these apply:\n'
            f'1. It is a pharmacopoeial reference standard (EP, USP, BP, IP standard) that only mentions '
            f'the molecule name — reference standards are analytical reagents, not the drug itself\n'
            f'2. It is an impurity standard, related compound, or degradation product marker\n'
            f'3. It is a different primary API/drug, and "{molecule}" appears only as a secondary reference '
            f'(e.g. "Spiramycin Ep Azithromycin Dihydrate Usp" — Spiramycin is the primary drug)\n'
            f'4. The molecule name is used as a calibrator, comparator substance, or test reference within a different product\n\n'
            f'Respond ONLY in JSON (no markdown, no extra text):\n'
            f'{{"is_relevant": <true|false>, "outlier_flag": <true|false>, '
            f'"outlier_reason": "<one sentence, empty string if relevant>"}}'
        )
        text = _llm_generate(prompt)
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            result = {
                "is_relevant": bool(parsed.get("is_relevant", True)),
                "outlier_flag": bool(parsed.get("outlier_flag", False)),
                "outlier_reason": str(parsed.get("outlier_reason", "")),
            }
        else:
            result = _relevant
    except Exception:
        result = _relevant

    _item_relevance_cache[cache_key] = result
    return result


def llm_filter_item_relevance(
    df: pd.DataFrame, molecule_name: str, item_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter rows where the item column does not directly represent the molecule.
    Iterates over unique item values (batched) to avoid redundant LLM calls.
    Returns: (relevant_df, item_outlier_df)
    item_outlier_df has columns: outlier_reason_item, outlier_reason_text,
    outlier_reason_qty, outlier_reason_price (for schema compatibility).
    """
    if df.empty or item_col not in df.columns:
        return df, pd.DataFrame()

    unique_items = df[item_col].dropna().unique()
    item_relevance: Dict[str, dict] = {}
    for item_val in unique_items:
        item_relevance[item_val] = llm_check_item_relevance(molecule_name, str(item_val))

    def _is_outlier(item_val):
        if pd.isna(item_val):
            return False
        return item_relevance.get(item_val, {}).get("outlier_flag", False)

    outlier_mask = df[item_col].apply(_is_outlier)
    relevant_df = df[~outlier_mask].copy()
    item_outlier_df = df[outlier_mask].copy()

    if len(item_outlier_df) > 0:
        item_outlier_df['outlier_reason_item'] = item_outlier_df[item_col].apply(
            lambda v: item_relevance.get(v, {}).get("outlier_reason", "") if not pd.isna(v) else ""
        )
        item_outlier_df['outlier_reason_text'] = item_outlier_df['outlier_reason_item']
        item_outlier_df['outlier_reason_qty'] = False
        item_outlier_df['outlier_reason_price'] = False

    return relevant_df, item_outlier_df


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


def llm_normalize_cipla_grade_spec(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize grade_spec column in Cipla GRN data using Gemini.
    Known mapping: 'no set pharmacopoeia present (IH can be used -inhouse)' -> 'IH'
    Falls back to a simple replace if Gemini is unavailable.
    """
    if 'grade_spec' not in df.columns:
        return df

    # Always apply the known hard-coded fix first (fast, no LLM needed)
    df['grade_spec'] = df['grade_spec'].replace(
        'no set pharmacopoeia present (IH can be used -inhouse)', 'IH'
    )

    if not _LLM_AVAILABLE:
        return df

    valid_grades = {"USP", "EP", "IP", "IH", "BP"}
    non_standard_mask = ~df['grade_spec'].str.upper().isin(valid_grades)
    non_standard_values = df.loc[non_standard_mask, 'grade_spec'].unique()

    if len(non_standard_values) == 0:
        return df

    try:
        values_list = "\n".join(f"- {v}" for v in non_standard_values)
        prompt = (
            "You are a pharmaceutical data normalizer. "
            "Map each of the following non-standard grade_spec values to one of: USP, EP, IP, IH, BP. "
            "Return a JSON object mapping each original value to its normalized grade. "
            "Example: {\"some value\": \"IH\"}.\n"
            f"Values to normalize:\n{values_list}"
        )
        text = _llm_generate(prompt)
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            mapping = json.loads(json_match.group())
            for orig, normalized in mapping.items():
                if normalized.upper() in valid_grades:
                    df['grade_spec'] = df['grade_spec'].replace(orig, normalized.upper())
    except Exception:
        pass  # fallback: already applied hard-coded fix above

    return df


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

    filter_stats = {
        'original_count': original_count,
        'filtered_count': filtered_count,
        'removed_count': removed,
        'removal_percentage': (removed / original_count * 100) if original_count > 0 else 0,
        'min_qty_threshold': min_qty,
        'price_lower': price_lower,
        'price_upper': price_upper,
    }
    outlier_df = llm_explain_outliers(outlier_df, filter_stats)
    return df, outlier_df, filter_stats


def llm_explain_outliers(outlier_df: pd.DataFrame, filter_stats: dict) -> pd.DataFrame:
    """
    Use Gemini to add a human-readable 'outlier_reason_text' column to the outlier DataFrame.
    Falls back to a simple rule-based reason if Gemini is unavailable.
    """
    if len(outlier_df) == 0:
        return outlier_df

    df = outlier_df.copy()

    # Rule-based fallback reason (always computed)
    min_qty = filter_stats.get('min_qty_threshold', 0)
    price_lower = filter_stats.get('price_lower', 0)
    price_upper = filter_stats.get('price_upper', float('inf'))

    def _rule_reason(row):
        reasons = []
        if row.get('outlier_reason_item'):
            reasons.append(
                f"Item '{row.get('ITEM', row.get('item', ''))}' is not a direct form of the molecule: "
                f"{row.get('outlier_reason_item', '')}"
            )
        if row.get('outlier_reason_qty', False):
            reasons.append(f"Quantity {row.get('QTY', 0):.0f} is below minimum threshold {min_qty:.0f}")
        if row.get('outlier_reason_price', False):
            reasons.append(f"Unit price ₹{row.get('unit_price', 0):.2f} is outside range ₹{price_lower:.0f}–₹{price_upper:.0f}")
        return " | ".join(reasons) if reasons else "Flagged as outlier"

    df['outlier_reason_text'] = df.apply(_rule_reason, axis=1)

    if not _LLM_AVAILABLE:
        return df

    try:
        # Only send a sample to the LLM to avoid token limits
        sample_df = df.head(5)
        sample = sample_df[['outlier_reason_text']].to_dict(orient='records')
        prompt = (
            "You are a pharmaceutical procurement analyst. "
            "Rewrite each of these technical outlier reasons into a clear, concise business explanation (max 15 words each). "
            "Return a JSON array of strings in the same order.\n"
            f"Reasons: {sample}"
        )
        text = _llm_generate(prompt)
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            explanations = json.loads(json_match.group())
            for i, explanation in enumerate(explanations[:len(sample_df)]):
                df.at[sample_df.index[i], 'outlier_reason_text'] = explanation
    except Exception:
        pass  # fallback reason already set above

    return df


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
        df['GRADE_SPEC'] = df[item_col].apply(llm_extract_grade_spec)
    else:
        df['GRADE_SPEC'] = 'USP'

    # Normalize UOM column
    if 'UQC' in df.columns:
        df = llm_normalize_uom(df, 'UQC')

    # Ensure QTY is numeric
    df['QTY'] = pd.to_numeric(df['QTY'], errors='coerce')
    df['TOTAL_VALUE'] = pd.to_numeric(df['TOTAL_VALUE'], errors='coerce')

    # Remove rows with missing critical values
    df = df.dropna(subset=['yyyymm', 'QTY', 'TOTAL_VALUE'])

    return df


def prepare_cipla_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare Cipla GRN data"""
    df = df.copy()

    df = llm_normalize_cipla_grade_spec(df)

    # Normalize UOM column
    if 'base_unit_of_measure' in df.columns:
        df = llm_normalize_uom(df, 'base_unit_of_measure')

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


def discover_export_files(molecule: str, data_dir: str, molecule_mapping: Dict) -> List[str]:
    """
    Discover export files for a given molecule using 'export_file_patterns'.

    Args:
        molecule: Molecule name (e.g., 'azithromycin')
        data_dir: Path to the data directory
        molecule_mapping: Molecule mapping config dict

    Returns:
        List of matching export file paths, or empty list if none found.
    """
    if molecule not in molecule_mapping['molecules']:
        return []

    patterns = molecule_mapping['molecules'][molecule].get('export_file_patterns', [])
    if not patterns:
        return []

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


def prepare_export_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare export file data.

    Export files have columns (case-insensitive):
      - Date           → yyyymm (format: 15-May-25)
      - product name   → ITEM
      - quantity       → QTY
      - unit           → UQC
      - FOB(INR)       → TOTAL_VALUE  (actual total transaction value, Free On Board in INR)

    Returns a clean DataFrame with columns: yyyymm, ITEM, QTY, UQC, TOTAL_VALUE, GRADE_SPEC
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Build a lower-cased column lookup for case-insensitive matching
    col_map = {c.strip().lower(): c for c in df.columns}

    def _find_col(*candidates):
        for cand in candidates:
            if cand.lower() in col_map:
                return col_map[cand.lower()]
        return None

    date_src = _find_col('date')
    item_src = _find_col('product name', 'product_name')
    qty_src = _find_col('quantity', 'qty')
    unit_src = _find_col('unit', 'uqc')
    value_src = _find_col('fob(inr)', 'fob (inr)', 'fob_inr', 'fob_(inr)', 'fob value inr', 'fob value (inr)')

    if date_src is None:
        raise ValueError(f"No Date column found in export file. Available columns: {list(df.columns)}")

    # Parse date: format '15-May-25'
    df['yyyymm'] = pd.to_datetime(df[date_src], format='%d-%b-%y', dayfirst=True, errors='coerce').dt.strftime('%Y%m')

    # Map columns
    df['ITEM'] = df[item_src] if item_src else ''
    df['QTY'] = pd.to_numeric(df[qty_src], errors='coerce') if qty_src else np.nan
    df['UQC'] = df[unit_src] if unit_src else ''
    df['TOTAL_VALUE'] = pd.to_numeric(df[value_src], errors='coerce') if value_src else np.nan

    # Extract grade spec from product name
    df['GRADE_SPEC'] = df['ITEM'].apply(llm_extract_grade_spec)

    # Normalize UOM
    df = llm_normalize_uom(df, 'UQC')

    # Drop rows with missing critical values
    df = df.dropna(subset=['yyyymm', 'QTY', 'TOTAL_VALUE'])

    return df[['yyyymm', 'ITEM', 'QTY', 'UQC', 'TOTAL_VALUE', 'GRADE_SPEC']].reset_index(drop=True)


def calculate_export_avg_price(export_df: pd.DataFrame) -> float:
    """
    Calculate the overall weighted average price from export data.

    Returns TOTAL_VALUE.sum() / QTY.sum(), or 0.0 if empty or QTY sum is 0.
    """
    if export_df.empty:
        return 0.0
    total_qty = export_df['QTY'].sum()
    if total_qty == 0:
        return 0.0
    return float(export_df['TOTAL_VALUE'].sum() / total_qty)


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


def llm_pipeline_summary(result: dict) -> str:
    """
    Use Gemini to generate a human-readable summary of the pipeline result.
    Returns empty string if Gemini is unavailable or result is not successful.
    """
    if not _LLM_AVAILABLE:
        return ""
    if result.get('status') != 'success':
        return ""
    try:
        meta = result.get('metadata', {})
        prompt = (
            "You are a pharmaceutical procurement analyst. "
            "Summarize this pipeline result in 2-3 sentences for a business audience. "
            "Be concise and focus on data quality and record counts.\n"
            f"Raw records: {meta.get('raw_record_count', 0)}\n"
            f"Filter stats: {meta.get('filter_stats', {})}\n"
            f"Cipla baseline avg price: ₹{meta.get('cipla_baseline', {}).get('avg_price', 0):,.2f}\n"
        )
        return _llm_generate(prompt)
    except Exception:
        return ""


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

        # Step 3b: Filter item-irrelevant rows (LLM item relevance check)
        item_col = next(
            (c for c in ['ITEM', 'item', 'ITEM_DESC', 'DESCRIPTION', 'PRODUCT'] if c in molecule_df.columns),
            None,
        )
        if item_col:
            molecule_df, item_outlier_df = llm_filter_item_relevance(molecule_df, molecule_name, item_col)
        else:
            item_outlier_df = pd.DataFrame()

        # Step 4: Calculate baselines
        cipla_baseline = calculate_cipla_baseline(cipla_df)

        # Step 5: Filter outliers (returns tuple of (filtered_df, outlier_df, stats))
        molecule_df_filtered, outlier_df, filter_stats = apply_outlier_filters(molecule_df, cipla_baseline)

        # Merge item-irrelevant outliers into the main outlier dataframe
        if len(item_outlier_df) > 0:
            outlier_df = pd.concat([item_outlier_df, outlier_df], ignore_index=True)

        filter_stats['item_outlier_count'] = len(item_outlier_df)

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
            },
            'llm_summary': llm_pipeline_summary({
                'status': 'success',
                'metadata': {
                    'raw_record_count': raw_record_count,
                    'filter_stats': filter_stats,
                    'cipla_baseline': cipla_baseline,
                }
            })
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
