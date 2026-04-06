# streamlit_app.py
import calendar
import math
import os
import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

try:
    import google.generativeai as genai
    _GEMINI_API_KEY = (
        st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None
    ) or os.environ.get("GEMINI_API_KEY")
    if _GEMINI_API_KEY:
        genai.configure(api_key=_GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        _GEMINI_AVAILABLE = True
    else:
        _GEMINI_AVAILABLE = False
        _gemini_model = None
except Exception:
    _GEMINI_AVAILABLE = False
    _gemini_model = None

from backend import (
    MOLECULE_MAPPING,
    load_cipla_grn, load_multiple_files,
    extract_grade_spec, extract_yyyymm,
    apply_outlier_filters, prepare_molecule_data, prepare_cipla_data,
    calculate_cipla_baseline, aggregate_supplier, aggregate_buyer, aggregate_cipla,
    discover_molecule_files, discover_cipla_file, get_available_molecules, get_molecule_file_info,
    match_molecule_input, get_suggestions, get_top_match, get_aliases,
    run_processing_pipeline,
    format_currency, format_percentage, calculate_price_variance,
    get_grade_spec_options, get_uom_options, get_date_range, filter_dataframe,
)

# LLM PANEL 3 — Supplier Price Analytics
# The rest of the original file with the 3 targeted changes applied

