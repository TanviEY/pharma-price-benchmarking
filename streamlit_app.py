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

# LLM is handled entirely by backend (Gemini → OpenAI fallback)

from backend import (
    MOLECULE_MAPPING,
    load_cipla_grn, load_multiple_files,
    extract_grade_spec, extract_yyyymm,
    apply_outlier_filters, prepare_molecule_data, prepare_cipla_data, llm_filter_item_relevance,
    calculate_cipla_baseline, aggregate_supplier, aggregate_buyer, aggregate_cipla,
    discover_molecule_files, discover_cipla_file, get_available_molecules, get_molecule_file_info,
    match_molecule_input, get_suggestions, get_top_match, get_aliases,
    run_processing_pipeline,
    format_currency, format_percentage, calculate_price_variance,
    get_grade_spec_options, get_uom_options, get_date_range, filter_dataframe,
    discover_export_files, prepare_export_data,
    _llm_generate, _LLM_AVAILABLE, _LLM_PROVIDER,
)


# ─── helper functions ────────────────────────────────────────────────────────

def _html(html_str: str) -> None:
    """Render HTML safely: strip leading whitespace per line to avoid Markdown
    code-block detection (Streamlit treats lines with 4+ leading spaces as code).
    """
    cleaned = "\n".join(line.lstrip() for line in html_str.split("\n") if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)

def _safe_wtd_avg(total_value_series, qty_series) -> float:
    total_qty = qty_series.sum()
    if total_qty == 0:
        return 0.0
    return total_value_series.sum() / total_qty


def yyyymm_to_label(yyyymm: str) -> str:
    """Convert '202504' → 'Apr 2025'"""
    try:
        y, m = int(str(yyyymm)[:4]), int(str(yyyymm)[4:])
        return f"{calendar.month_abbr[m]} {y}"
    except Exception:
        return str(yyyymm)


def fmt_inr(value: float) -> str:
    """Format as ₹X.XX L or ₹X.XX Cr"""
    if value >= 1e7:
        return f"₹{value/1e7:.2f} Cr"
    elif value >= 1e5:
        return f"₹{value/1e5:.2f} L"
    else:
        return f"₹{value:,.0f}"


def fmt_qty(value: float) -> str:
    return f"{int(value):,}"


def _gemini_narrate(step_description: str, step_result_summary: str) -> str:
    """Narrate a pipeline step using the active LLM. Returns '' on failure."""
    if not _LLM_AVAILABLE:
        return ""
    prompt = (
        "You are a data processing agent. Narrate this step in ONE sentence "
        "(max 20 words), present tense, no technical jargon:\n"
        f"Step: {step_description}\n"
        f"Result: {step_result_summary}"
    )
    return _llm_generate(prompt)


def _llm_analysis(prompt: str) -> str:
    """Call the active LLM with a prompt and return the response text. Returns '' on failure."""
    if not _LLM_AVAILABLE:
        return ""
    return _llm_generate(prompt)


AVATAR_COLORS = [
    "#3b82f6", "#16a34a", "#7c3aed", "#d97706",
    "#0891b2", "#dc2626", "#0d9488",
]


def _avatar_color(idx: int) -> str:
    return AVATAR_COLORS[idx % len(AVATAR_COLORS)]


def _initials(name: str) -> str:
    parts = name.strip().split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[1][0]).upper()
    return (name[:2]).upper() if len(name) >= 2 else (name.upper() or "??")


def _render_sparkline(monthly_values, color="#3b82f6") -> str:
    if not monthly_values or all(v == 0 for v in monthly_values):
        return ""
    max_v = max(monthly_values) if max(monthly_values) > 0 else 1
    pos_vals = [v for v in monthly_values if v > 0]
    min_v = min(pos_vals) if pos_vals else 0
    bars = ""
    for v in monthly_values:
        if max_v > min_v and v > 0:
            pct = (v - min_v) / (max_v - min_v)
        elif v > 0:
            pct = 0.5
        else:
            pct = 0.05
        h = int(5 + pct * 19)
        bars += (
            f'<div style="display:inline-block;width:5px;height:{h}px;'
            f'background:{color};margin:0 1px;border-radius:1px;'
            f'vertical-align:bottom;opacity:0.8;"></div>'
        )
    return (
        f'<div style="display:flex;align-items:flex-end;gap:1px;'
        f'margin-top:6px;height:26px;">{bars}</div>'
    )


def _pagination_bar(total_items, page_size, current_page, state_key, prefix):
    """Renders numbered page buttons. Returns the (possibly updated) current page."""
    total_pages = max(1, math.ceil(total_items / page_size))
    if total_pages <= 1:
        return 1
    cols = st.columns(min(total_pages, _MAX_PAGE_BTNS))
    for i, col in enumerate(cols):
        page_num = i + 1
        label = f"**{page_num}**" if page_num == current_page else str(page_num)
        with col:
            btn_style = "primary" if page_num == current_page else "secondary"
            if st.button(label, key=f"{prefix}_pg_{page_num}", type=btn_style):
                st.session_state[state_key] = page_num
                st.rerun()
    return current_page


_PAGE_SIZE = 10       # rows per page for all paginated tables / charts
_MAX_PAGE_BTNS = 10   # max numbered page buttons to display at once

# LLM analysis panel thresholds
_PARITY_THRESHOLD = 0.02          # ±2% = price parity band (Panel 1)
_GROWTH_THRESHOLD = 5.0           # avg MoM % change > 5% → Growing (Panel 3)
_DECLINE_THRESHOLD = -5.0         # avg MoM % change < -5% → Declining (Panel 3)
_VOL_INCREASE_THRESHOLD = 20.0    # % MoM avg volume growth → "Increasing" (Panel 4)
_VOL_DECREASE_THRESHOLD = -20.0   # % MoM avg volume change → "Decreasing" (Panel 4)


# ─── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PharmaIntel · Price Benchmarking",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
  --bg:#f4f6fb; --white:#ffffff; --border:#e4e9f2; --subtle:#f0f4ff;
  --blue:#3b82f6; --blue-dk:#1d4ed8; --blue-lt:#eff6ff;
  --cyan:#0891b2; --teal:#0d9488; --teal-lt:#f0fdfa;
  --green:#16a34a; --green-lt:#f0fdf4;
  --amber:#d97706; --amber-lt:#fffbeb;
  --red:#dc2626; --red-lt:#fff1f2;
  --t1:#0f172a; --t2:#334155; --t3:#64748b; --t4:#94a3b8;
}

/* Hide Streamlit defaults */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="stSidebar"] {display: none;}
.block-container {
  padding-top: 0 !important;
  padding-left: 0 !important;
  padding-right: 0 !important;
  max-width: 100% !important;
}
[data-testid="stAppViewContainer"] { background: var(--bg); }
div[data-testid="stVerticalBlockBorderWrapper"] { padding: 0 !important; }

/* ── NAV BAR ── */
.pi-navbar {
  background: #ffffff;
  box-shadow: 0 1px 6px rgba(0,0,0,0.10);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.6rem 2rem;
  position: sticky;
  top: 0;
  z-index: 999;
}
.pi-brand-name { font-size: 1.2rem; font-weight: 800; color: var(--t1); }
.pi-brand-name span { color: var(--blue); }
.pi-brand-sub { font-size: 0.72rem; color: var(--t3); margin-left: 0.5rem; }
.pi-nav-right { display: flex; align-items: center; gap: 0.5rem; }
.pi-pill-live {
  background:#f0fdf4; color:#16a34a; border:1px solid #bbf7d0;
  padding:3px 10px; border-radius:20px; font-size:0.72rem; font-weight:600;
}
.pi-pill-cipla {
  background:var(--blue-lt); color:var(--blue-dk); border:1px solid #bfdbfe;
  padding:3px 10px; border-radius:20px; font-size:0.72rem; font-weight:600;
}
.pi-pill-fy {
  background:var(--subtle); color:var(--t2); border:1px solid var(--border);
  padding:3px 10px; border-radius:20px; font-size:0.72rem; font-weight:500;
}
.pi-avatar {
  width:32px; height:32px; border-radius:50%;
  background:linear-gradient(135deg,var(--blue),var(--blue-dk));
  color:#fff; display:inline-flex; align-items:center; justify-content:center;
  font-size:0.72rem; font-weight:700;
}

/* ── HERO ── */
.pi-hero {
  background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 55%, #1d4ed8 100%);
  padding: 2.2rem 2rem 1.8rem 2rem;
}
.pi-hero-title {
  font-size: 1.9rem; font-weight: 800; color: #ffffff; margin-bottom: 0.4rem;
}
.pi-hero-title span {
  background: linear-gradient(90deg, #60a5fa, #38bdf8);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
.pi-hero-sub { font-size: 0.88rem; color: rgba(255,255,255,0.7); margin-bottom: 0; }

/* Search card */
.pi-search-wrap {
  background: #ffffff;
  border-radius: 14px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.18);
  padding: 1rem 1.5rem 0.5rem 1.5rem;
  margin: 1.5rem 1.5rem 1rem 1.5rem;
  position: relative;
  z-index: 10;
}
.pi-search-wrap label { font-size: 0.75rem !important; color: var(--t3) !important; font-weight: 600 !important; }
.pi-search-wrap input[type="text"] {
  border: none !important; border-bottom: 2px solid var(--border) !important;
  border-radius: 0 !important; box-shadow: none !important;
  font-size: 0.95rem !important;
}
.pi-search-wrap [data-baseweb="select"] > div {
  border: none !important; border-bottom: 2px solid var(--border) !important;
  border-radius: 0 !important; box-shadow: none !important;
}
.pi-search-wrap .pi-analyse-btn > button {
  background: linear-gradient(90deg, #1d4ed8, #0891b2) !important;
  color: #fff !important;
  font-weight: 700 !important;
  border: none !important;
  border-radius: 8px !important;
  width: 100% !important;
  margin-top: 0.75rem !important;
  padding: 0.55rem 1rem !important;
  font-size: 0.9rem !important;
  letter-spacing: 0.3px !important;
}
.pi-search-wrap .pi-analyse-btn > button:hover { opacity: 0.9; }

/* ── SUGGESTION PANEL ── */
.pi-suggestion-panel {
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.6rem 1rem;
  margin: 0 1.5rem 0.75rem 1.5rem;
  box-shadow: 0 4px 16px rgba(0,0,0,0.08);
}
.pi-suggestion-label {
  font-size: 0.72rem;
  color: var(--t3);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 0.4rem;
}
.pi-suggestion-panel .stButton > button {
  background: var(--subtle) !important;
  color: var(--t2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 20px !important;
  padding: 3px 14px !important;
  font-size: 0.8rem !important;
  font-weight: 500 !important;
  margin: 2px !important;
}
.pi-suggestion-panel .stButton > button:hover {
  background: var(--blue-lt) !important;
  color: var(--blue-dk) !important;
  border-color: #bfdbfe !important;
}

/* Month filter row above Section 2 */
.pi-month-filter [data-baseweb="select"] > div {
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  font-size: 0.78rem !important;
  background: #ffffff !important;
}

/* Recent tag pills (Streamlit buttons styled as pills) */
.pi-tag-row { padding: 0.4rem 1.5rem 0.6rem 1.5rem; }
.pi-tag-row .stButton > button {
  background: rgba(255,255,255,0.15) !important;
  color: #ffffff !important;
  border: 1px solid rgba(255,255,255,0.35) !important;
  border-radius: 20px !important;
  padding: 3px 14px !important;
  font-size: 0.78rem !important;
  font-weight: 500 !important;
  margin-bottom: 4px !important;
}
.pi-tag-row .stButton > button:hover {
  background: rgba(255,255,255,0.25) !important;
}

/* ── MATERIAL BANNER ── */
.pi-mat-banner {
  background: #ffffff;
  border-radius: 12px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.07);
  padding: 1rem 1.5rem;
  margin: 1rem 1.5rem 0 1.5rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
}
.pi-mat-icon {
  width:48px; height:48px; border-radius:12px;
  background:linear-gradient(135deg,var(--blue),var(--blue-dk));
  display:inline-flex; align-items:center; justify-content:center;
  font-size:1.4rem; flex-shrink:0;
}
.pi-mat-name { font-size:1.1rem; font-weight:700; color:var(--t1); }
.pi-chip {
  display:inline-flex; align-items:center;
  background:var(--subtle); color:var(--t2);
  padding:2px 10px; border-radius:20px;
  font-size:0.72rem; font-weight:500; margin:2px;
}

/* Export buttons (Streamlit) */
.pi-export-row .stButton > button,
.pi-export-row .stDownloadButton > button {
  border: 1.5px solid var(--border) !important;
  background: #ffffff !important; color: var(--t2) !important;
  border-radius: 8px !important; font-size: 0.78rem !important;
  font-weight: 600 !important; padding: 0.35rem 1rem !important;
}
.pi-export-row .stButton > button:hover,
.pi-export-row .stDownloadButton > button:hover {
  background: var(--subtle) !important;
}

/* ── KPI CARDS ── */
.pi-kpi-card {
  background: #ffffff; border-radius: 12px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.06);
  padding: 1rem 1.1rem; border-top: 3px solid var(--blue); height: 100%;
}
.pi-kpi-label {
  font-size: 0.67rem; font-weight: 700; color: var(--t3);
  text-transform: uppercase; letter-spacing: 0.7px; margin-bottom: 0.4rem;
}
.pi-kpi-value { font-size: 1.4rem; font-weight: 800; color: var(--t1); line-height: 1.1; }
.pi-kpi-value span { font-size: 0.75rem; font-weight: 400; color: var(--t3); }
.pi-kpi-badge {
  display:inline-flex; align-items:center;
  padding:2px 8px; border-radius:12px; font-size:0.68rem;
  font-weight:600; margin-top:5px;
}
.pi-kpi-note { font-size:0.67rem; color:var(--t4); margin-top:3px; }

/* ── CARD WRAPPER ── */
.pi-card {
  background: #ffffff; border-radius: 12px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.06);
  padding: 1.2rem 1.4rem;
}
.pi-section-hdr,
.pi-section-header {
  font-size:0.7rem; font-weight:700; color:var(--t3);
  text-transform:uppercase; letter-spacing:0.8px; margin-bottom:0.2rem;
}
.pi-section-title { font-size:1rem; font-weight:700; color:var(--t1); margin-bottom:0.2rem; }
.pi-section-sub { font-size:0.78rem; color:var(--t3); margin-bottom:1.25rem; }
.pi-card-content { padding-top: 1rem; }

/* ── HORIZONTAL BAR CHART ── */
.pi-bar-row { display:flex; align-items:center; gap:10px; margin-bottom:7px; }
.pi-bar-label {
  width:170px; font-size:0.78rem; color:var(--t1); font-weight:500;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; flex-shrink:0;
}
.pi-bar-label.cipla { font-weight:700; color:var(--blue-dk); }
.pi-bar-track {
  flex:1; height:26px; background:var(--subtle); border-radius:4px;
  overflow:hidden; position:relative;
}
.pi-bar-fill {
  height:100%; border-radius:4px; display:flex; align-items:center;
  font-size:0.75rem; font-weight:600;
  color:#fff; white-space:nowrap; min-width:40px;
}
.pi-bar-price { width:88px; font-size:0.8rem; font-weight:600; color:var(--t1); text-align:right; flex-shrink:0; }
.pi-bar-badge { width:78px; font-size:0.72rem; font-weight:600; flex-shrink:0; }
.pi-bar-scale {
  display:flex; justify-content:space-between; font-size:0.67rem;
  color:var(--t4); margin-top:3px; margin-left:180px;
}

/* ── COMPETITOR TABLE ── */
.pi-comp-table { width:100%; border-collapse:collapse; font-size:0.78rem; }
.pi-comp-table th {
  background:var(--subtle); color:var(--t3); padding:0.45rem 0.65rem;
  text-align:left; font-weight:600; font-size:0.67rem;
  text-transform:uppercase; letter-spacing:0.5px; border-bottom:2px solid var(--border);
}
.pi-comp-table td {
  padding:0.45rem 0.65rem; color:var(--t1);
  border-bottom:1px solid var(--border); vertical-align:middle;
}
.pi-comp-table tr.cipla-row td { background:var(--blue-lt) !important; }
.pi-comp-table tr:hover td { background:var(--subtle); }
.pi-av {
  width:30px; height:30px; border-radius:50%;
  display:inline-flex; align-items:center; justify-content:center;
  font-size:0.62rem; font-weight:700; color:#fff; flex-shrink:0;
}

/* ── DATA TABLES ── */
.pi-data-table { width:100%; border-collapse:collapse; font-size:0.78rem; }
.pi-data-table th {
  background:var(--subtle); color:var(--t3); padding:0.4rem 0.65rem;
  text-align:left; font-weight:600; font-size:0.67rem;
  text-transform:uppercase; letter-spacing:0.4px; border-bottom:2px solid var(--border);
}
.pi-data-table td {
  padding:0.4rem 0.65rem; color:var(--t1); border-bottom:1px solid var(--border);
}
.pi-data-table tr.footer-row td {
  font-weight:700; background:var(--subtle) !important; border-top:2px solid var(--border);
}
.pi-data-table tr:hover td { background:var(--subtle); }

/* ── BADGES ── */
.badge { display:inline-block; padding:2px 8px; border-radius:12px; font-size:0.67rem; font-weight:700; }
.badge-blue  { background:var(--blue-lt); color:var(--blue-dk); }
.badge-green { background:var(--green-lt); color:var(--green); }
.badge-amber { background:var(--amber-lt); color:var(--amber); }
.badge-red   { background:var(--red-lt); color:var(--red); }
.badge-cyan  { background:#ecfeff; color:var(--cyan); }

/* ── INFO BANNER ── */
.pi-info-banner {
  background:var(--blue-lt); border:1px solid #bfdbfe; color:var(--blue-dk);
  border-radius:10px; padding:0.75rem 1.2rem; font-size:0.85rem;
  margin-bottom:1rem;
}

/* ── FOOTER ── */
.pi-footer {
  background:#ffffff; border-top:1px solid var(--border);
  padding:1rem 2rem; display:flex; justify-content:space-between;
  align-items:center; font-size:0.72rem; color:var(--t3); margin-top:2rem;
}
.pi-footer a { color:var(--t3); text-decoration:none; margin-left:1rem; }
.pi-footer a:hover { color:var(--blue); }

/* page padding */
.pi-page-body { padding: 0 1.5rem; }

/* ── AGENT LOG BOX ── */
.pi-agent-log {
  background: #0d1117;
  border: 1px solid #30363d;
  border-radius: 10px;
  padding: 1rem 1.2rem;
  margin: 1rem 1.5rem;
  font-family: 'Courier New', Courier, monospace;
  font-size: 0.82rem;
  max-height: 320px;
  overflow-y: auto;
  line-height: 1.7;
}
.pi-log-step-ok   { color: #3fb950; }
.pi-log-step-warn { color: #d29922; }
.pi-log-step-err  { color: #f85149; }
.pi-log-narration { color: #8b949e; font-style: italic; margin-left: 1.5rem; }
.pi-log-header    { color: #58a6ff; font-weight: 700; margin-bottom: 0.4rem; }

/* ── FILTER PANEL ── */
.pi-filter-panel {
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: 12px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.06);
  padding: 1.2rem 1.5rem 1rem 1.5rem;
  margin: 1rem 1.5rem;
}
.pi-filter-title {
  font-size: 0.95rem; font-weight: 700; color: var(--t1); margin-bottom: 0.25rem;
}
.pi-filter-note {
  font-size: 0.72rem; color: var(--t3); margin-bottom: 0.8rem; font-style: italic;
}
</style>
""", unsafe_allow_html=True)

# ─── session state ────────────────────────────────────────────────────────────
for _k, _v in [
    ("selected_molecule", None),
    ("pipeline_result", None),
    ("pipeline_clean_time", None),
    ("chart_month_filter", "All Months"),
    ("bar_view_mode", "Top 25% by Volume"),
    ("bar_page", 1),
    ("comp_table_page", 1),
    ("cipla_table_page", 1),
    ("exim_table_page", 1),
    ("bargain_page", 1),
    ("filter_from_month", None),
    ("filter_to_month", None),
    ("filter_uoms", None),
    ("filter_grades", None),
    ("filters_applied", False),
    ("llm_buyer_trend_cache", {}),
    ("llm_bargain_cache", {}),
    ("llm_supplier_vol_cache", {}),
    ("llm_supplier_vol_shift_cache", {}),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _on_mol_enter():
    st.session_state["_analyse_trigger"] = True

# ─── init objects ─────────────────────────────────────────────────────────────
available_molecules = get_available_molecules("data/raw", MOLECULE_MAPPING)

# ─── NAV BAR ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="pi-navbar">
  <div style="display:flex;align-items:center;gap:0.5rem;">
    <span style="font-size:1.4rem;">💊</span>
    <span class="pi-brand-name">Pharma<span>Intel</span></span>
    <span class="pi-brand-sub">Price Benchmarking</span>
  </div>
  <div class="pi-nav-right">
    <span class="pi-pill-live">● Live</span>
    <span class="pi-pill-cipla">Cipla Internal</span>
    <span class="pi-avatar">CI</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── HERO SEARCH ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="pi-hero">
  <div class="pi-hero-title">Material <span>Price Benchmarking</span></div>
  <div class="pi-hero-sub">
    Intelligent procurement analysis across EXIM trade data and internal ERP
  </div>
</div>
""", unsafe_allow_html=True)

# Search card (placed below the hero with a clean positive margin)
with st.container():
    st.markdown('<div class="pi-search-wrap">', unsafe_allow_html=True)
    sc1, sc_btn = st.columns([5, 2])
    with sc1:
        hero_mol_input = st.text_input(
            "Molecule",
            value=st.session_state.selected_molecule or "",
            placeholder="e.g., Azithromycin…",
            key="hero_mol_input",
            on_change=_on_mol_enter,
        )
    with sc_btn:
        st.markdown('<div class="pi-analyse-btn">', unsafe_allow_html=True)
        analyse_clicked = st.button("Analyse", key="hero_analyse_btn")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Check Enter-key trigger
_enter_triggered = st.session_state.pop("_analyse_trigger", False)

# Suggestion panel (shown when there is typed input that differs from selected molecule)
_show_suggestions = (
    hero_mol_input.strip()
    and hero_mol_input.strip().lower() != (st.session_state.selected_molecule or "").lower()
)
if _show_suggestions:
    suggestions = get_suggestions(hero_mol_input.strip(), MOLECULE_MAPPING, top_n=5)
    all_zero = all(s == 0 for _, s in suggestions)
    if suggestions:
        st.markdown('<div class="pi-suggestion-panel">', unsafe_allow_html=True)
        if all_zero:
            st.markdown('<div class="pi-suggestion-label">No close match · Showing available molecules</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="pi-suggestion-label">Did you mean?</div>', unsafe_allow_html=True)
        sug_cols = st.columns(len(suggestions))
        for i, (mol_name, score) in enumerate(suggestions):
            with sug_cols[i]:
                suggestion_button_label = f"{mol_name.upper()} {score}%" if score > 0 else mol_name.upper()
                if st.button(suggestion_button_label, key=f"sug_{mol_name}_{i}"):
                    st.session_state.selected_molecule = mol_name
                    st.session_state.pipeline_result = None
                    st.session_state.filters_applied = False
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# Handle Analyse click or Enter key
if (analyse_clicked or _enter_triggered) and hero_mol_input.strip():
    top_match = get_top_match(hero_mol_input.strip(), MOLECULE_MAPPING)
    if top_match and top_match in MOLECULE_MAPPING["molecules"]:
        st.session_state.selected_molecule = top_match
        st.session_state.pipeline_result = None
        st.session_state.filters_applied = False
        st.rerun()
    else:
        st.markdown(
            f'<div class="pi-info-banner">⚠️ No matching molecule found for '
            f'<strong>{hero_mol_input}</strong>. Please check the spelling.</div>',
            unsafe_allow_html=True,
        )

# ─── MAIN CONTENT ─────────────────────────────────────────────────────────────
if st.session_state.selected_molecule:
    selected_mol = st.session_state.selected_molecule

    # ── run pipeline (step-by-step with Gemini narration) ─────────────────────
    if st.session_state.pipeline_result is None:
        _log_lines: list = []
        _log_placeholder = st.empty()

        def _render_log(lines):
            inner = "\n".join(lines)
            _log_placeholder.markdown(
                f'<div class="pi-agent-log">'
                f'<div class="pi-log-header">⚙ PharmaIntel · Agent Processing — {selected_mol.upper()}</div>'
                f'{inner}'
                f'</div>',
                unsafe_allow_html=True,
            )

        _t0 = time.time()

        try:
            # ── Step 1: Discover files ─────────────────────────────────────────
            _log_lines.append('<span class="pi-log-step-ok">▶ Step 1/8 — Discovering files…</span>')
            _render_log(_log_lines)
            mol_files = discover_molecule_files(selected_mol, "data/raw", MOLECULE_MAPPING)
            cipla_file = discover_cipla_file("data/raw")
            if not mol_files:
                st.markdown(
                    f'<div class="pi-info-banner">❌ No data files found for <strong>{selected_mol}</strong>.</div>',
                    unsafe_allow_html=True,
                )
                st.stop()
            if not cipla_file:
                st.markdown(
                    '<div class="pi-info-banner">❌ Cipla GRN file not found.</div>',
                    unsafe_allow_html=True,
                )
                st.stop()
            _s1_result = f"{len(mol_files)} EXIM file(s) found, Cipla GRN located"
            _narr1 = _gemini_narrate(
                f"Discovering files for {selected_mol}",
                _s1_result,
            )
            _log_lines.append(f'<span class="pi-log-step-ok">✔ Step 1 done — {_s1_result}</span>')
            if _narr1:
                _log_lines.append(f'<span class="pi-log-narration">💬 {_narr1}</span>')
            _render_log(_log_lines)

            # ── Step 2: Load data ──────────────────────────────────────────────
            _log_lines.append(f'<span class="pi-log-step-ok">▶ Step 2/8 — Loading {len(mol_files)} EXIM files + Cipla GRN file…</span>')
            _render_log(_log_lines)
            molecule_df_raw = load_multiple_files(mol_files)
            api_filter = MOLECULE_MAPPING["molecules"][selected_mol]["cipla_api_filter"]
            cipla_df_raw = load_cipla_grn(cipla_file, api_filter)
            raw_record_count = len(molecule_df_raw)
            _s2_result = f"{raw_record_count} raw EXIM rows + {len(cipla_df_raw)} Cipla rows loaded"
            _narr2 = _gemini_narrate(
                f"Loading {len(mol_files)} EXIM files + Cipla GRN file",
                _s2_result,
            )
            _log_lines.append(f'<span class="pi-log-step-ok">✔ Step 2 done — {_s2_result}</span>')
            if _narr2:
                _log_lines.append(f'<span class="pi-log-narration">💬 {_narr2}</span>')
            _render_log(_log_lines)

            # ── Step 2b: Discover and load export files (non-blocking) ────────
            export_files = discover_export_files(selected_mol, "data/raw", MOLECULE_MAPPING)
            export_df_raw = load_multiple_files(export_files) if export_files else pd.DataFrame()
            if export_files:
                _log_lines.append(f'<span class="pi-log-step-ok">  ↳ Export files found: {len(export_files)} file(s)</span>')
                _render_log(_log_lines)

            # ── Step 3: Prepare molecule data ──────────────────────────────────
            _log_lines.append('<span class="pi-log-step-ok">▶ Step 3/8 — Preparing &amp; cleaning molecule data…</span>')
            _render_log(_log_lines)
            molecule_df = prepare_molecule_data(molecule_df_raw)
            after_clean_count = len(molecule_df)
            _s3_result = f"{raw_record_count - after_clean_count} rows dropped (nulls/bad dates), {after_clean_count} remaining"
            _narr3 = _gemini_narrate(
                "Preparing & cleaning molecule data (drop nulls, parse dates, extract grade spec)",
                _s3_result,
            )
            _log_lines.append(f'<span class="pi-log-step-ok">✔ Step 3 done — {_s3_result}</span>')
            if _narr3:
                _log_lines.append(f'<span class="pi-log-narration">💬 {_narr3}</span>')
            _render_log(_log_lines)

            # ── Step 3b: Prepare export data ───────────────────────────────────
            try:
                export_df = prepare_export_data(export_df_raw) if not export_df_raw.empty else pd.DataFrame()
            except Exception as _exp_exc:
                _log_lines.append(f'<span class="pi-log-step-warn">  ↳ Export data skipped — {_exp_exc}</span>')
                export_df = pd.DataFrame()

            # ── Step 3c: LLM item-relevance filter ────────────────────────────
            _item_col = next(
                (c for c in ["ITEM", "item", "ITEM_DESC", "DESCRIPTION", "PRODUCT"] if c in molecule_df.columns),
                None,
            )
            if _item_col and _LLM_AVAILABLE:
                _log_lines.append('<span class="pi-log-step-ok">▶ Step 3c — LLM item-relevance check…</span>')
                _render_log(_log_lines)
                molecule_df, _item_outlier_df = llm_filter_item_relevance(molecule_df, selected_mol, _item_col)
                _item_removed = len(_item_outlier_df)
                _log_lines.append(
                    f'<span class="pi-log-step-ok">✔ Step 3c done — {_item_removed} item-irrelevant rows flagged as outliers</span>'
                )
                _render_log(_log_lines)
            else:
                _item_outlier_df = pd.DataFrame()

            # ── Step 4: Prepare Cipla data ─────────────────────────────────────
            _log_lines.append('<span class="pi-log-step-ok">▶ Step 4/8 — Preparing Cipla data…</span>')
            _render_log(_log_lines)
            cipla_df = prepare_cipla_data(cipla_df_raw)
            _s4_result = f"{len(cipla_df)} Cipla rows cleaned and ready"
            _narr4 = _gemini_narrate("Preparing Cipla data", _s4_result)
            _log_lines.append(f'<span class="pi-log-step-ok">✔ Step 4 done — {_s4_result}</span>')
            if _narr4:
                _log_lines.append(f'<span class="pi-log-narration">💬 {_narr4}</span>')
            _render_log(_log_lines)

            # ── Step 5: Calculate Cipla baseline ──────────────────────────────
            _log_lines.append('<span class="pi-log-step-ok">▶ Step 5/8 — Calculating Cipla price baseline…</span>')
            _render_log(_log_lines)
            cipla_baseline = calculate_cipla_baseline(cipla_df)
            _avg_price = cipla_baseline["avg_price"]
            _s5_result = f"avg price ₹{_avg_price:,.2f}/unit, {cipla_baseline['total_records']} records"
            _narr5 = _gemini_narrate(
                f"Calculating Cipla price baseline (avg price: ₹{_avg_price:.2f}/unit)",
                _s5_result,
            )
            _log_lines.append(f'<span class="pi-log-step-ok">✔ Step 5 done — {_s5_result}</span>')
            if _narr5:
                _log_lines.append(f'<span class="pi-log-narration">💬 {_narr5}</span>')
            _render_log(_log_lines)

            # ── Step 6: Apply outlier filters ──────────────────────────────────
            _log_lines.append('<span class="pi-log-step-ok">▶ Step 6/8 — Applying outlier filters (qty threshold + price ±30%)…</span>')
            _render_log(_log_lines)
            molecule_df_filtered, outlier_df, filter_stats = apply_outlier_filters(molecule_df, cipla_baseline)
            # Merge LLM item-relevance outliers into the outlier table
            if len(_item_outlier_df) > 0:
                outlier_df = pd.concat([_item_outlier_df, outlier_df], ignore_index=True)
            _removed = filter_stats["removed_count"]
            _pct = filter_stats["removal_percentage"]
            _s6_result = f"{_removed} outlier rows removed ({_pct:.1f}%), {filter_stats['filtered_count']} rows kept"
            _narr6 = _gemini_narrate(
                "Applying outlier filters (qty threshold + price ±30%)",
                _s6_result,
            )
            _log_lines.append(f'<span class="pi-log-step-ok">✔ Step 6 done — {_s6_result}</span>')
            if _narr6:
                _log_lines.append(f'<span class="pi-log-narration">💬 {_narr6}</span>')
            _render_log(_log_lines)

            # ── Step 7: Aggregate ──────────────────────────────────────────────
            _log_lines.append('<span class="pi-log-step-ok">▶ Step 7/8 — Aggregating by Supplier, Buyer, Cipla…</span>')
            _render_log(_log_lines)
            supplier_agg = aggregate_supplier(molecule_df_filtered)
            buyer_agg = aggregate_buyer(molecule_df_filtered)
            cipla_agg = aggregate_cipla(cipla_df, selected_mol)
            _s7_result = (
                f"{len(supplier_agg)} supplier rows, "
                f"{len(buyer_agg)} buyer rows, "
                f"{len(cipla_agg)} Cipla rows"
            )
            _narr7 = _gemini_narrate("Aggregating by Supplier, Buyer, Cipla", _s7_result)
            _log_lines.append(f'<span class="pi-log-step-ok">✔ Step 7 done — {_s7_result}</span>')
            if _narr7:
                _log_lines.append(f'<span class="pi-log-narration">💬 {_narr7}</span>')
            _render_log(_log_lines)

            # ── Step 8: Build consolidated dataset ────────────────────────────
            _log_lines.append('<span class="pi-log-step-ok">▶ Step 8/8 — Building consolidated dataset…</span>')
            _render_log(_log_lines)
            shared_cols = [
                "entity_name", "yyyymm", "uom", "GRADE_SPEC",
                "Sum_of_QTY", "Sum_of_TOTAL_VALUE", "Avg_PRICE", "source",
            ]
            supplier_agg["entity_name"] = supplier_agg["supplier"]
            buyer_agg["entity_name"] = buyer_agg["buyer"]
            cipla_agg["entity_name"] = cipla_agg["api"]
            consolidated = pd.concat(
                [supplier_agg[shared_cols], buyer_agg[shared_cols], cipla_agg[shared_cols]],
                ignore_index=True,
            )
            # Save processed files (non-blocking — warn if a file is locked e.g. open in Excel)
            _proc_dir = Path("data/processed")
            _proc_dir.mkdir(parents=True, exist_ok=True)
            _save_errors = []
            for _fname, _df in [
                (f"{selected_mol}_supplier.csv", supplier_agg),
                (f"{selected_mol}_buyer.csv", buyer_agg),
                (f"cipla_{selected_mol}.csv", cipla_agg),
                (f"outlier_{selected_mol}.csv", outlier_df),
            ]:
                try:
                    _df.to_csv(_proc_dir / _fname, index=False)
                except PermissionError:
                    _save_errors.append(_fname)
            if _save_errors:
                _log_lines.append(
                    f'<span class="pi-log-step-warn">⚠ Could not save {len(_save_errors)} file(s) — '
                    f'close them in Excel and re-run if needed: {", ".join(_save_errors)}</span>'
                )

            _t1 = time.time()
            _clean_time = _t1 - _t0
            _s8_result = f"{len(consolidated)} consolidated rows ready in {_clean_time:.1f}s"
            _narr8 = _gemini_narrate("Building consolidated dataset", _s8_result)
            _log_lines.append(f'<span class="pi-log-step-ok">✔ Step 8 done — {_s8_result}</span>')
            if _narr8:
                _log_lines.append(f'<span class="pi-log-narration">💬 {_narr8}</span>')
            _log_lines.append(f'<span class="pi-log-step-ok" style="font-weight:700;">✅ Pipeline complete — {len(consolidated)} records ready for analysis</span>')
            _render_log(_log_lines)

            # Cache the result
            st.session_state.pipeline_result = {
                "status": "success",
                "errors": [],
                "metadata": {
                    "files_loaded": mol_files,
                    "raw_record_count": raw_record_count,
                    "after_clean_count": after_clean_count,
                    "filter_stats": filter_stats,
                    "cipla_baseline": cipla_baseline,
                },
                "data": {
                    "supplier": supplier_agg,
                    "buyer": buyer_agg,
                    "cipla": cipla_agg,
                    "consolidated": consolidated,
                    "outlier": outlier_df,
                    "export": export_df,
                },
            }
            st.session_state.pipeline_clean_time = _clean_time

        except Exception as _exc:
            _log_lines.append(f'<span class="pi-log-step-err">❌ Pipeline error: {_exc}</span>')
            _render_log(_log_lines)
            st.stop()

    # ── load cached result ─────────────────────────────────────────────────────
    result = st.session_state.pipeline_result
    if result is None or result["status"] == "failed":
        st.markdown(
            f'<div class="pi-info-banner">❌ Pipeline failed: '
            f'{", ".join((result or {}).get("errors", ["Unknown error"]))}</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    consolidated_df = result["data"]["consolidated"]
    outlier_df_cached = result["data"]["outlier"]
    export_df_cached = result["data"].get("export", pd.DataFrame())
    meta = result["metadata"]
    filter_stats_cached = meta["filter_stats"]
    cipla_baseline_cached = meta["cipla_baseline"]

    # Metadata (derived from full consolidated_df)
    mol_cfg = MOLECULE_MAPPING["molecules"].get(selected_mol, {})
    cas_code = mol_cfg.get("cipla_api_filter", selected_mol.upper())
    import_duty_pct = mol_cfg.get("import_duty_pct", 0)
    import_duty_mult = 1 + import_duty_pct / 100
    uom = consolidated_df["uom"].mode()[0] if len(consolidated_df) > 0 else "KG"
    grade_series = consolidated_df[consolidated_df["source"] == "Cipla"]["GRADE_SPEC"]
    grade = grade_series.mode()[0] if len(grade_series) > 0 else "USP"

    # Build month/UOM/grade options from full consolidated_df
    available_months_raw = sorted(consolidated_df["yyyymm"].unique())
    available_months_labels = [yyyymm_to_label(m) for m in available_months_raw]
    month_label_to_yyyymm = {yyyymm_to_label(m): m for m in available_months_raw}

    # ── CLEANING SUMMARY EXPANDER ──────────────────────────────────────────────
    with st.expander("📋 Cleaning Summary", expanded=False):
        _ec1, _ec2, _ec3, _ec4 = st.columns(4)
        raw_ct = meta.get("raw_record_count", 0)
        clean_ct = meta.get("after_clean_count", raw_ct)
        final_ct = filter_stats_cached.get("filtered_count", clean_ct)
        removed_pct = filter_stats_cached.get("removal_percentage", 0.0)
        _clean_t = st.session_state.get("pipeline_clean_time") or 0.0
        with _ec1:
            st.metric("Raw Records", f"{raw_ct:,}")
        with _ec2:
            st.metric("After Null/Date Cleaning", f"{clean_ct:,}")
        with _ec3:
            st.metric("After Outlier Removal", f"{final_ct:,}")
        with _ec4:
            st.metric("% Removed (Outliers)", f"{removed_pct:.1f}%")

        st.caption(f"⏱ Pipeline completed in **{_clean_t:.2f} seconds**")

        if len(outlier_df_cached) > 0:
            st.markdown("**Outlier Details** — rows removed and reasons:")
            # Use actual thresholds stored by apply_outlier_filters
            _min_qty_thr = filter_stats_cached.get("min_qty_threshold", 0.0)
            _price_lower = filter_stats_cached.get("price_lower", cipla_baseline_cached["avg_price"] * 0.70)
            _price_upper = filter_stats_cached.get("price_upper", cipla_baseline_cached["avg_price"] * 1.30)

            def _outlier_reason(row):
                reasons = []
                if row.get("outlier_reason_qty", False):
                    reasons.append(f"Low Quantity (QTY < {_min_qty_thr:.0f})")
                if row.get("outlier_reason_price", False):
                    reasons.append(f"Price Out of Range (₹{_price_lower:.0f}–₹{_price_upper:.0f})")
                if row.get("outlier_reason_item"):
                    reasons.append(f"Irrelevant Item: {row['outlier_reason_item']}")
                return " | ".join(reasons) if reasons else "Unknown"

            outlier_display = outlier_df_cached.copy()
            outlier_display["Reason"] = outlier_display.apply(_outlier_reason, axis=1)

            # Find entity name column
            _ent_col = None
            for _c in ["Supp_Name", "IMPORTER", "entity_name"]:
                if _c in outlier_display.columns:
                    _ent_col = _c
                    break

            _disp_cols = {}
            if _ent_col:
                _disp_cols[_ent_col] = "Supplier / Importer"
            if "yyyymm" in outlier_display.columns:
                _disp_cols["yyyymm"] = "Date (yyyymm)"
            # Show item description for item-relevance outliers
            for _icol in ["ITEM", "item", "ITEM_DESC", "DESCRIPTION", "PRODUCT"]:
                if _icol in outlier_display.columns:
                    _disp_cols[_icol] = "Item Description"
                    break
            if "QTY" in outlier_display.columns:
                _disp_cols["QTY"] = "QTY"
            if "unit_price" in outlier_display.columns:
                _disp_cols["unit_price"] = "Unit Price (₹)"
            _disp_cols["Reason"] = "Reason"

            _show_cols = [c for c in _disp_cols if c in outlier_display.columns]
            _show_df = (
                outlier_display[_show_cols]
                .rename(columns=_disp_cols)
                .reset_index(drop=True)
            )

            # Build per-column config so wide columns (Reason, Item, Supplier) get
            # enough space and text is never clipped.
            _col_cfg = {}
            for _col_label in _show_df.columns:
                if _col_label in ("Supplier / Importer", "Item Description"):
                    _col_cfg[_col_label] = st.column_config.TextColumn(_col_label, width="large")
                elif _col_label == "Reason":
                    _col_cfg[_col_label] = st.column_config.TextColumn(_col_label, width="large")
                elif _col_label in ("QTY", "Unit Price (₹)", "Date (yyyymm)"):
                    _col_cfg[_col_label] = st.column_config.TextColumn(_col_label, width="small")

            _tbl_height = min(400, 35 + len(_show_df) * 35)
            st.dataframe(
                _show_df,
                use_container_width=True,
                hide_index=True,
                height=_tbl_height,
                column_config=_col_cfg,
            )
        else:
            st.success("✅ No outliers detected.")

    # ── FILTER PANEL ──────────────────────────────────────────────────────────
    st.markdown('<div class="pi-filter-panel">', unsafe_allow_html=True)
    st.markdown('<div class="pi-filter-title">🔎 Configure Analysis Filters</div>', unsafe_allow_html=True)
    st.markdown('<div class="pi-filter-note">Pipeline cached — changing filters does not re-run data processing.</div>', unsafe_allow_html=True)

    _all_uoms = sorted(consolidated_df["uom"].dropna().unique().tolist())
    _all_grades = sorted(consolidated_df["GRADE_SPEC"].dropna().unique().tolist())

    # Defaults: initialise from session_state or use full range
    _def_from = st.session_state.get("filter_from_month")
    _def_to = st.session_state.get("filter_to_month")
    _def_uoms = st.session_state.get("filter_uoms")
    _def_grades = st.session_state.get("filter_grades")

    if _def_from not in available_months_labels or _def_from is None:
        _def_from = available_months_labels[0] if available_months_labels else None
    if _def_to not in available_months_labels or _def_to is None:
        _def_to = available_months_labels[-1] if available_months_labels else None
    if _def_uoms is None:
        _def_uoms = _all_uoms
    if _def_grades is None:
        _def_grades = _all_grades

    fp1, fp2, fp3, fp4 = st.columns([2, 2, 3, 3])
    with fp1:
        _from_sel = st.selectbox(
            "From Month",
            available_months_labels,
            index=available_months_labels.index(_def_from) if _def_from in available_months_labels else 0,
            key="fp_from_month",
        )
    with fp2:
        _to_sel = st.selectbox(
            "To Month",
            available_months_labels,
            index=available_months_labels.index(_def_to) if _def_to in available_months_labels else max(0, len(available_months_labels) - 1),
            key="fp_to_month",
        )
    with fp3:
        _uoms_sel = st.multiselect(
            "UOM",
            _all_uoms,
            default=[u for u in _def_uoms if u in _all_uoms],
            key="fp_uoms",
        )
    with fp4:
        _grades_sel = st.multiselect(
            "Grade Spec",
            _all_grades,
            default=[g for g in _def_grades if g in _all_grades],
            key="fp_grades",
        )

    _, _apply_col = st.columns([6, 1])
    with _apply_col:
        _apply_clicked = st.button("Apply Filters", key="fp_apply", type="primary")

    st.markdown("</div>", unsafe_allow_html=True)  # pi-filter-panel

    if _apply_clicked:
        st.session_state["filter_from_month"] = _from_sel
        st.session_state["filter_to_month"] = _to_sel
        st.session_state["filter_uoms"] = _uoms_sel if _uoms_sel else _all_uoms
        st.session_state["filter_grades"] = _grades_sel if _grades_sel else _all_grades
        st.session_state["filters_applied"] = True
        # Reset paginations when filter changes
        for _pk in ["bar_page", "comp_table_page", "cipla_table_page", "exim_table_page", "bargain_page"]:
            st.session_state[_pk] = 1
        st.rerun()


    if not st.session_state.get("filters_applied", False):
        st.markdown("""
        <div style="text-align:center; padding: 2.5rem 2rem; margin: 1rem 1.5rem;">
          <div style="font-size:2.5rem; margin-bottom:0.75rem;">🎛️</div>
          <div style="font-size:1.1rem; font-weight:700; color:#0f172a; margin-bottom:0.4rem;">
            Configure your filters to begin analysis
          </div>
          <div style="font-size:0.88rem; color:#64748b; max-width:480px; margin:0 auto;">
            Select the date range, UOM and Grade Spec above, then click
            <strong>Apply Filters</strong> to load the visualizations.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Apply filter values to consolidated_df ─────────────────────────────────
        _f_from_yyyymm = month_label_to_yyyymm.get(
            st.session_state.get("filter_from_month") or available_months_labels[0] if available_months_labels else "",
            available_months_raw[0] if available_months_raw else "",
        )
        _f_to_yyyymm = month_label_to_yyyymm.get(
            st.session_state.get("filter_to_month") or available_months_labels[-1] if available_months_labels else "",
            available_months_raw[-1] if available_months_raw else "",
        )
        _f_uoms = st.session_state.get("filter_uoms") or _all_uoms
        _f_grades = st.session_state.get("filter_grades") or _all_grades

        filtered_df = consolidated_df[
            (consolidated_df["yyyymm"] >= _f_from_yyyymm)
            & (consolidated_df["yyyymm"] <= _f_to_yyyymm)
            & (consolidated_df["uom"].isin(_f_uoms))
            & (consolidated_df["GRADE_SPEC"].isin(_f_grades))
        ]

        # Context label for chart subtitles
        _from_lbl = yyyymm_to_label(_f_from_yyyymm) if _f_from_yyyymm else "—"
        _to_lbl = yyyymm_to_label(_f_to_yyyymm) if _f_to_yyyymm else "—"
        month_context = f"{_from_lbl} – {_to_lbl}" if _from_lbl != _to_lbl else _from_lbl

        # ── MATERIAL BANNER ──────────────────────────────────────────────────────
        export_csv = consolidated_df.to_csv(index=False).encode("utf-8")

        bann_l, bann_r = st.columns([5, 1])
        with bann_l:
            st.markdown(f"""
            <div class="pi-mat-banner">
              <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">
                <div class="pi-mat-icon">🧪</div>
                <div>
                  <div class="pi-mat-name">{selected_mol.upper()}</div>
                  <div style="margin-top:4px;">
                    <span class="pi-chip">API</span>
                    <span class="pi-chip">CAS {cas_code[:22]}</span>
                    <span class="pi-chip">INR / {uom}</span>
                  </div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        with bann_r:
            st.markdown(
                '<div class="pi-export-row" style="display:flex;gap:6px;justify-content:flex-end;margin-top:1rem;margin-right:1.5rem;">',
                unsafe_allow_html=True,
            )
            st.download_button(
                label="Export CSV",
                data=export_csv,
                file_name=f"{selected_mol}_all.csv",
                mime="text/csv",
                key="excel_export",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # ─────────────────────────────────────────────────────────────────────────
        # SECTION 1 — 6 KPI Cards with Sparklines
        # ─────────────────────────────────────────────────────────────────────────
        cipla_df_f = filtered_df[filtered_df["source"] == "Cipla"]
        market_df_f = filtered_df[filtered_df["source"] == "Buyer"]
        buyer_df_f = filtered_df[filtered_df["source"] == "Buyer"]

        cipla_price = _safe_wtd_avg(cipla_df_f["Sum_of_TOTAL_VALUE"], cipla_df_f["Sum_of_QTY"])
        market_price = _safe_wtd_avg(market_df_f["Sum_of_TOTAL_VALUE"], market_df_f["Sum_of_QTY"]) * import_duty_mult
        cipla_n_records = len(cipla_df_f)
        cipla_total_qty = cipla_df_f["Sum_of_QTY"].sum()
        market_n_ent = market_df_f["entity_name"].nunique()

        # Per-entity WTD avg (buyer-only for Lowest/Highest Competitor cards)
        low_price = high_price = 0.0
        low_ent = high_ent = "—"
        if len(buyer_df_f) > 0:
            ent_prices = (
                buyer_df_f.groupby("entity_name")
                .apply(lambda g: _safe_wtd_avg(g["Sum_of_TOTAL_VALUE"], g["Sum_of_QTY"]))
                .reset_index(name="wtd_price")
            )
            ent_prices = ent_prices[ent_prices["wtd_price"] > 0]
            if len(ent_prices) > 0:
                min_row = ent_prices.loc[ent_prices["wtd_price"].idxmin()]
                max_row = ent_prices.loc[ent_prices["wtd_price"].idxmax()]
                low_price, low_ent = min_row["wtd_price"] * import_duty_mult, min_row["entity_name"]
                high_price, high_ent = max_row["wtd_price"] * import_duty_mult, max_row["entity_name"]

        cost_adv = cipla_price - market_price if market_price > 0 else 0.0
        cost_pct = abs(cost_adv / market_price * 100) if market_price > 0 else 0.0

        # Sparklines — monthly WTD avg across ALL months (consolidated_df, not filtered)
        months_sorted_all = sorted(consolidated_df["yyyymm"].unique())

        def _monthly_wtd(src_fn, months):
            vals = []
            for m in months:
                mdf = consolidated_df[consolidated_df["yyyymm"] == m]
                mdf = src_fn(mdf)
                vals.append(_safe_wtd_avg(mdf["Sum_of_TOTAL_VALUE"], mdf["Sum_of_QTY"]))
            return vals

        cipla_spark = _render_sparkline(
            _monthly_wtd(lambda d: d[d["source"] == "Cipla"], months_sorted_all), "#3b82f6"
        )
        market_spark = _render_sparkline(
            _monthly_wtd(lambda d: d[d["source"] == "Buyer"], months_sorted_all), "#0891b2"
        )

        # Export sparkline — monthly weighted avg from export_df_cached with active filters
        if not export_df_cached.empty:
            _export_mask = (
                (export_df_cached["UQC"].isin(_f_uoms))
                & (export_df_cached["GRADE_SPEC"].isin(_f_grades))
            )
            if _f_from_yyyymm and _f_to_yyyymm:
                _export_mask = (
                    _export_mask
                    & (export_df_cached["yyyymm"] >= _f_from_yyyymm)
                    & (export_df_cached["yyyymm"] <= _f_to_yyyymm)
                )
            _export_filtered = export_df_cached[_export_mask]
            export_avg_price = _safe_wtd_avg(_export_filtered["TOTAL_VALUE"], _export_filtered["QTY"])
            _export_by_month = {
                m: grp for m, grp in _export_filtered.groupby("yyyymm")
            }
            _export_spark_vals = [
                _safe_wtd_avg(_export_by_month[_m]["TOTAL_VALUE"], _export_by_month[_m]["QTY"])
                if _m in _export_by_month else 0.0
                for _m in months_sorted_all
            ]
        else:
            export_avg_price = 0.0
            _export_spark_vals = [0.0] * len(months_sorted_all)
        export_spark = _render_sparkline(_export_spark_vals, "#7c3aed")

        # Cost advantage badge
        if cipla_price > 0 and market_price > 0:
            adv_sym = "▼" if cost_adv < 0 else "▲"
            adv_word = "below" if cost_adv < 0 else "above"
            adv_text = f"{adv_sym} {cost_pct:.1f}% {adv_word} market"
        else:
            adv_text = month_context

        st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="pi-page-body">', unsafe_allow_html=True)
        k1, k2, k_export, k3, k4, k5 = st.columns(6)

        with k1:
            st.markdown(f"""
            <div class="pi-kpi-card" style="border-top-color:#3b82f6;">
              <div class="pi-kpi-label">Cipla WTD Avg · ERP</div>
              <div class="pi-kpi-value">₹{cipla_price:,.0f} <span>/{uom}</span></div>
              <div><span class="pi-kpi-badge" style="background:#eff6ff;color:#1d4ed8;">{month_context}</span></div>
              <div class="pi-kpi-note">{cipla_n_records} POs · {fmt_qty(cipla_total_qty)} {uom}</div>
              {cipla_spark}
            </div>
            """, unsafe_allow_html=True)

        with k2:
            st.markdown(f"""
            <div class="pi-kpi-card" style="border-top-color:#0891b2;">
              <div class="pi-kpi-label">EXIM Market Avg (Import)</div>
              <div class="pi-kpi-value">₹{market_price:,.0f} <span>/{uom}</span></div>
              <div><span class="pi-kpi-badge" style="background:#ecfeff;color:#0891b2;">{market_n_ent} competitors</span></div>
              <div class="pi-kpi-note">EXIM import data · incl. {import_duty_pct}% duty</div>
              {market_spark}
            </div>
            """, unsafe_allow_html=True)

        with k_export:
            st.markdown(f"""
            <div class="pi-kpi-card" style="border-top-color:#7c3aed;">
              <div class="pi-kpi-label">EXIM Market Avg (Export)</div>
              <div class="pi-kpi-value">₹{export_avg_price:,.0f} <span>/{uom}</span></div>
              <div><span class="pi-kpi-badge" style="background:#f5f3ff;color:#7c3aed;">EXIM export data</span></div>
              <div class="pi-kpi-note">Weighted avg across period</div>
              {export_spark}
            </div>
            """, unsafe_allow_html=True)

        with k3:
            adv_bg = "#f0fdf4" if cost_adv <= 0 else "#fff1f2"
            adv_fg = "#16a34a" if cost_adv <= 0 else "#dc2626"
            adv_sign = "−" if cost_adv < 0 else "+"
            st.markdown(f"""
            <div class="pi-kpi-card" style="border-top-color:#16a34a;">
              <div class="pi-kpi-label">Cost Advantage</div>
              <div class="pi-kpi-value" style="color:{adv_fg};">{adv_sign}₹{abs(cost_adv):,.0f} <span>/{uom}</span></div>
              <div><span class="pi-kpi-badge" style="background:{adv_bg};color:{adv_fg};">{adv_text}</span></div>
              <div class="pi-kpi-note">vs EXIM avg</div>
            </div>
            """, unsafe_allow_html=True)

        with k4:
            st.markdown(f"""
            <div class="pi-kpi-card" style="border-top-color:#d97706;">
              <div class="pi-kpi-label">Lowest Competitor</div>
              <div class="pi-kpi-value">₹{low_price:,.0f} <span>/{uom}</span></div>
              <div><span class="pi-kpi-badge" style="background:#fffbeb;color:#d97706;">{low_ent[:22] if low_ent != "—" else "—"}</span></div>
              <div class="pi-kpi-note">period avg · incl. {import_duty_pct}% duty</div>
            </div>
            """, unsafe_allow_html=True)

        with k5:
            st.markdown(f"""
            <div class="pi-kpi-card" style="border-top-color:#dc2626;">
              <div class="pi-kpi-label">Highest Competitor</div>
              <div class="pi-kpi-value">₹{high_price:,.0f} <span>/{uom}</span></div>
              <div><span class="pi-kpi-badge" style="background:#fff1f2;color:#dc2626;">{high_ent[:22] if high_ent != "—" else "—"}</span></div>
              <div class="pi-kpi-note">premium grade · incl. {import_duty_pct}% duty</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # pi-page-body
        st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)

        # ─────────────────────────────────────────────────────────────────────────
        # LLM PANEL 1 — Buyer Price Trend Analysis
        # ─────────────────────────────────────────────────────────────────────────
        _p1_cipla_df = filtered_df[filtered_df["source"] == "Cipla"]
        _p1_buyer_df = filtered_df[filtered_df["source"] == "Buyer"]

        _p1_empty = len(_p1_cipla_df) == 0 and len(_p1_buyer_df) == 0

        _html("""
        <div class="pi-page-body">
        <div class="pi-card">
        <div class="pi-section-header">BUYER PRICE TREND ANALYSIS</div>
        <div class="pi-section-title">Cipla Avg vs EXIM Market Avg — Monthly Trend</div>
        """ + f'<div class="pi-section-sub">Buyer perspective · {month_context} · {uom}</div>' + '<div class="pi-card-content">')

        if _p1_empty:
            _html('<div class="pi-info-banner">No data available for the selected filters.</div>')
        else:
            # Build monthly weighted avg per source
            _p1_months = sorted(filtered_df["yyyymm"].unique())
            _p1_cipla_monthly = {}
            _p1_buyer_monthly = {}
            for _m in _p1_months:
                _cm = _p1_cipla_df[_p1_cipla_df["yyyymm"] == _m]
                _p1_cipla_monthly[_m] = _safe_wtd_avg(_cm["Sum_of_TOTAL_VALUE"], _cm["Sum_of_QTY"])
                _bm = _p1_buyer_df[_p1_buyer_df["yyyymm"] == _m]
                _p1_buyer_monthly[_m] = _safe_wtd_avg(_bm["Sum_of_TOTAL_VALUE"], _bm["Sum_of_QTY"])

            _p1_labels = [yyyymm_to_label(m) for m in _p1_months]
            _p1_cipla_vals = [_p1_cipla_monthly[m] for m in _p1_months]
            _p1_buyer_vals = [_p1_buyer_monthly[m] for m in _p1_months]

            # Build Plotly figure
            _p1_fig = go.Figure()
            _p1_fig.add_trace(go.Scatter(
                x=_p1_labels, y=_p1_cipla_vals,
                mode="lines+markers", name="Cipla Avg Price",
                line=dict(color="#3b82f6", dash="dash"), marker=dict(size=6),
            ))
            _p1_fig.add_trace(go.Scatter(
                x=_p1_labels, y=_p1_buyer_vals,
                mode="lines+markers", name="EXIM Market Avg (Buyer)",
                line=dict(color="#0891b2"), marker=dict(size=6),
            ))
            _p1_fig.update_layout(
                height=340,
                paper_bgcolor="white", plot_bgcolor="white",
                margin=dict(l=40, r=20, t=20, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(
                    title=f"Price (₹/{uom})",
                    tickformat=",.0f",
                    nticks=15,
                    tickmode="auto",
                    showgrid=True,
                    gridcolor="#e4e9f2",
                    gridwidth=1,
                    griddash="dot",
                    autorange=True,
                ),
                xaxis=dict(
                    tickangle=-30,
                ),
            )
            st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
            st.plotly_chart(_p1_fig, use_container_width=True)

            # %Higher / %Lower / %Parity analytics
            _p1_compare_months = [
                m for m in _p1_months
                if _p1_cipla_monthly[m] > 0 and _p1_buyer_monthly[m] > 0
            ]
            _p1_total = len(_p1_compare_months)
            if _p1_total > 0:
                _p1_higher_count = sum(
                    1 for m in _p1_compare_months
                    if _p1_buyer_monthly[m] > _p1_cipla_monthly[m]
                )
                _p1_lower_count = sum(
                    1 for m in _p1_compare_months
                    if _p1_buyer_monthly[m] < _p1_cipla_monthly[m]
                )
                _p1_parity_count = sum(
                    1 for m in _p1_compare_months
                    if abs(_p1_buyer_monthly[m] - _p1_cipla_monthly[m]) / _p1_cipla_monthly[m] <= _PARITY_THRESHOLD
                )
                pct_higher = round(_p1_higher_count / _p1_total * 100, 1)
                pct_lower = round(_p1_lower_count / _p1_total * 100, 1)
                pct_parity = round(_p1_parity_count / _p1_total * 100, 1)
            else:
                pct_higher = pct_lower = pct_parity = 0.0

            _html(f"""
            <div style="display:flex;gap:0.75rem;flex-wrap:wrap;margin-top:1rem;margin-bottom:0.5rem;">
            <span class="badge-red">📈 Higher {pct_higher}% of months</span>
            <span class="badge-green">📉 Lower {pct_lower}% of months</span>
            <span class="badge-amber">≈ Parity {pct_parity}% of months</span>
            </div>
            """)

            # LLM Narrative
            _p1_cache_key = (selected_mol, _f_from_yyyymm, _f_to_yyyymm)
            _p1_cache = st.session_state.get("llm_buyer_trend_cache", {})
            if _p1_cache_key in _p1_cache:
                _p1_llm_text = _p1_cache[_p1_cache_key]
            else:
                _p1_cipla_dict = {yyyymm_to_label(m): round(_p1_cipla_monthly[m], 2) for m in _p1_months}
                _p1_buyer_dict = {yyyymm_to_label(m): round(_p1_buyer_monthly[m], 2) for m in _p1_months}
                _p1_prompt = (
                    f"You are a pharmaceutical procurement analyst. In 3-4 sentences, describe the price trend "
                    f"from the buyer's perspective for {selected_mol}. "
                    f"Monthly Cipla avg prices: {_p1_cipla_dict}. "
                    f"Monthly EXIM buyer avg prices: {_p1_buyer_dict}. "
                    f"In {pct_higher}% of months, buyers paid more than Cipla; in {pct_lower}% they paid less; "
                    f"in {pct_parity}% prices were at parity (within 2%). "
                    f"Period: {month_context}. "
                    f"Highlight whether buyers are generally paying more or less than Cipla's internal price, "
                    f"note any notable months or turning points, and give one actionable recommendation."
                )
                _p1_llm_text = _llm_analysis(_p1_prompt)
                _p1_cache[_p1_cache_key] = _p1_llm_text
                st.session_state["llm_buyer_trend_cache"] = _p1_cache

            if _p1_llm_text:
                _html(f"""
                <div style="background:#f0f9ff;border-left:3px solid #0891b2;border-radius:8px;padding:0.75rem 1rem;margin-top:1rem;font-size:0.85rem;color:#0f172a;line-height:1.6;">
                <span style="font-weight:700;">🤖 AI Analysis · </span>{_p1_llm_text}
                </div>
                """)
            else:
                # Fallback rule-based text
                if pct_lower >= pct_higher:
                    _p1_fallback = (
                        f"Buyers are generally purchasing {selected_mol} below Cipla's internal benchmark price "
                        f"({pct_lower}% of months), suggesting a competitive external market."
                    )
                else:
                    _p1_fallback = (
                        f"Buyers are paying above Cipla's internal benchmark in {pct_higher}% of the selected months, "
                        f"indicating Cipla has a cost advantage in external procurement."
                    )
                _html(f"""
                <div style="background:#f0f9ff;border-left:3px solid #0891b2;border-radius:8px;padding:0.75rem 1rem;margin-top:1rem;font-size:0.85rem;color:#0f172a;line-height:1.6;">
                <span style="font-weight:700;">📊 Analysis · </span>{_p1_fallback}
                </div>
                """)

        _html("</div></div></div>")  # pi-card-content, pi-card, pi-page-body
        _html('<div style="height:1rem;"></div>')

        # ─────────────────────────────────────────────────────────────────────────
        # LLM PANEL 2 — Bargain Buyers
        # ─────────────────────────────────────────────────────────────────────────
        _p2_cipla_df = filtered_df[filtered_df["source"] == "Cipla"]
        _p2_buyer_df = filtered_df[filtered_df["source"] == "Buyer"]
        _p2_cipla_price = _safe_wtd_avg(_p2_cipla_df["Sum_of_TOTAL_VALUE"], _p2_cipla_df["Sum_of_QTY"])
        _p2_bargain_threshold = _p2_cipla_price

        _html(f"""
        <div class="pi-page-body">
        <div class="pi-card">
        <div class="pi-section-header">BARGAIN BUYER ANALYSIS</div>
        <div class="pi-section-title">Buyers Purchasing at Below-Benchmark Prices</div>
        <div class="pi-section-sub">Identified buyers · {month_context} · threshold = Cipla avg price</div>
        <div class="pi-card-content">
        """)

        if len(_p2_buyer_df) == 0:
            _html('<div class="pi-info-banner">No buyer data available for the selected filters.</div>')
        else:
            # Group by entity_name
            _p2_grp = (
                _p2_buyer_df.groupby("entity_name")
                .apply(lambda g: pd.Series({
                    "wtd_price": _safe_wtd_avg(g["Sum_of_TOTAL_VALUE"], g["Sum_of_QTY"]),
                    "total_qty": g["Sum_of_QTY"].sum(),
                    "total_value": g["Sum_of_TOTAL_VALUE"].sum(),
                }))
                .reset_index()
            )
            _p2_grp = _p2_grp[_p2_grp["wtd_price"] > 0]
            _p2_bargain = _p2_grp[_p2_grp["wtd_price"] <= _p2_bargain_threshold].sort_values("wtd_price")
            _p2_total_buyers = len(_p2_grp)
            _p2_bargain_count = len(_p2_bargain)

            # Summary line
            _html(f"""
            <div style="font-size:0.85rem;color:#334155;margin-bottom:0.75rem;">
            <strong>{_p2_bargain_count}</strong> out of <strong>{_p2_total_buyers}</strong> buyers are purchasing
            below Cipla's internal price benchmark
            (threshold: ₹{_p2_bargain_threshold:,.0f}/{uom}).
            </div>
            """)

            if _p2_bargain_count == 0:
                _html('<div class="pi-info-banner">No buyers are purchasing below the bargain threshold in this period.</div>')
            else:
                # Pagination for bargain table
                _p2_page = st.session_state.get("bargain_page", 1)
                _p2_page_size = _PAGE_SIZE  # 10
                _p2_start = (_p2_page - 1) * _p2_page_size
                _p2_end = _p2_start + _p2_page_size
                _p2_bargain_page = _p2_bargain.iloc[_p2_start:_p2_end]

                # Build HTML table
                _p2_rows = ""
                for _, _row in _p2_bargain_page.iterrows():
                    _vs_cipla = ((_row["wtd_price"] - _p2_cipla_price) / _p2_cipla_price * 100) if _p2_cipla_price > 0 else 0
                    _p2_rows += f"""
                    <tr>
                    <td>{_row['entity_name']}</td>
                    <td>₹{_row['wtd_price']:,.0f}</td>
                    <td>{int(_row['total_qty']):,}</td>
                    <td style="color:#16a34a;">{_vs_cipla:+.1f}%</td>
                    <td><span class="badge-green">✅ Bargain</span></td>
                    </tr>
                    """
                _html(f"""
                <table class="pi-data-table" style="width:100%;border-collapse:collapse;margin-top:0.5rem;">
                <thead><tr>
                <th>Buyer</th><th>Avg Price</th><th>Volume ({uom})</th><th>% vs Cipla</th><th>Position</th>
                </tr></thead>
                <tbody>{_p2_rows}</tbody>
                </table>
                """)
                # Render pagination buttons inline (inside the panel)
                _pagination_bar(len(_p2_bargain), _p2_page_size, _p2_page, "bargain_page", "bargain")

            # LLM Narrative
            _p2_cache_key = (selected_mol, _f_from_yyyymm, _f_to_yyyymm)
            _p2_cache = st.session_state.get("llm_bargain_cache", {})
            if _p2_cache_key in _p2_cache:
                _p2_llm_text = _p2_cache[_p2_cache_key]
            else:
                if _p2_bargain_count > 0:
                    _p2_cheapest_row = _p2_bargain.iloc[0]
                    _p2_cheapest_buyer = _p2_cheapest_row["entity_name"]
                    _p2_cheapest_price = _p2_cheapest_row["wtd_price"]
                    _p2_bargain_list = [
                        {"buyer": r["entity_name"], "price": round(r["wtd_price"], 2), "qty": int(r["total_qty"])}
                        for _, r in _p2_bargain.iterrows()
                    ]
                    _p2_prompt = (
                        f"You are a pharmaceutical procurement analyst. "
                        f"{_p2_bargain_count} buyers are purchasing {selected_mol} below Cipla's internal benchmark "
                        f"of ₹{_p2_cipla_price:.0f}/{uom} in {month_context}. "
                        f"The cheapest buyer is {_p2_cheapest_buyer} at ₹{_p2_cheapest_price:.0f}/{uom}. "
                        f"Bargain buyers: {_p2_bargain_list}. "
                        f"In 3-4 sentences, explain what this means for Cipla's procurement position and "
                        f"give one concrete negotiation insight."
                    )
                    _p2_llm_text = _llm_analysis(_p2_prompt)
                else:
                    _p2_llm_text = ""
                _p2_cache[_p2_cache_key] = _p2_llm_text
                st.session_state["llm_bargain_cache"] = _p2_cache

            if _p2_llm_text:
                _html(f"""
                <div style="background:#f0f9ff;border-left:3px solid #0891b2;border-radius:8px;padding:0.75rem 1rem;margin-top:1rem;font-size:0.85rem;color:#0f172a;line-height:1.6;">
                <span style="font-weight:700;">🤖 AI Analysis · </span>{_p2_llm_text}
                </div>
                """)
            elif _p2_bargain_count == 0 and len(_p2_buyer_df) > 0:
                _html(f"""
                <div style="background:#f0f9ff;border-left:3px solid #0891b2;border-radius:8px;padding:0.75rem 1rem;margin-top:1rem;font-size:0.85rem;color:#0f172a;line-height:1.6;">
                <span style="font-weight:700;">📊 Analysis · </span>
                No buyers are purchasing {selected_mol} below the 5% discount threshold relative to Cipla's benchmark
                price of ₹{_p2_cipla_price:,.0f}/{uom} in {month_context}.
                This indicates Cipla's procurement price is competitive in the current market.
                </div>
                """)

        _html("</div></div></div>")  # pi-card-content, pi-card, pi-page-body
        _html('<div style="height:1rem;"></div>')

        # ─────────────────────────────────────────────────────────────────────────
        # LLM PANEL 3 — Supplier Avg Price Analytics
        # ─────────────────────────────────────────────────────────────────────────
        _p3_supplier_df = filtered_df[filtered_df["source"] == "Supplier"]

        _html(f"""
        <div class="pi-page-body">
        <div class="pi-card">
        <div class="pi-section-header">SUPPLIER AVG PRICE ANALYTICS</div>
        <div class="pi-section-title">Supplier Avg Price Trends — Monthly Comparison</div>
        <div class="pi-section-sub">Supplier perspective · {month_context} · ₹/{uom}</div>
        <div class="pi-card-content" style="margin-top:1rem;">
        """)

        if len(_p3_supplier_df) == 0:
            _html('<div class="pi-info-banner">No supplier data available for the selected filters.</div>')
        else:
            # Monthly volume by supplier
            _p3_months = sorted(_p3_supplier_df["yyyymm"].unique())
            _p3_labels = [yyyymm_to_label(m) for m in _p3_months]

            # Supplier totals
            _p3_totals = (
                _p3_supplier_df.groupby("entity_name")
                .apply(lambda g: pd.Series({
                    "total_qty": g["Sum_of_QTY"].sum(),
                    "wtd_price": _safe_wtd_avg(g["Sum_of_TOTAL_VALUE"], g["Sum_of_QTY"]),
                    "month_count": g["yyyymm"].nunique(),
                }))
                .reset_index()
                .sort_values("wtd_price", ascending=False)
            )

            # Top 5 suppliers by avg price for line chart
            _p3_top5 = _p3_totals.head(5)["entity_name"].tolist()

            # Compute MoM trend per supplier based on monthly avg price
            def _p3_supplier_trend(sup_name):
                _sd = _p3_supplier_df[_p3_supplier_df["entity_name"] == sup_name]
                _sm = (
                    _sd.groupby("yyyymm")
                    .apply(lambda g: _safe_wtd_avg(g["Sum_of_TOTAL_VALUE"], g["Sum_of_QTY"]))
                    .sort_index()
                )
                _sm = _sm[_sm > 0]
                if len(_sm) < 2:
                    return "Stable"
                _pct_changes = _sm.pct_change().dropna() * 100
                _avg_mom = _pct_changes.mean()
                if _avg_mom > _GROWTH_THRESHOLD:
                    return "Rising"
                elif _avg_mom < _DECLINE_THRESHOLD:
                    return "Falling"
                return "Stable"

            _p3_totals["trend"] = _p3_totals["entity_name"].apply(_p3_supplier_trend)

            # Line chart — monthly avg price per top-5 supplier
            _p3_fig = go.Figure()
            for _idx, _sup in enumerate(_p3_top5):
                _sup_monthly_price = []
                for _m in _p3_months:
                    _sdf = _p3_supplier_df[
                        (_p3_supplier_df["entity_name"] == _sup) & (_p3_supplier_df["yyyymm"] == _m)
                    ]
                    _sup_monthly_price.append(_safe_wtd_avg(_sdf["Sum_of_TOTAL_VALUE"], _sdf["Sum_of_QTY"]))
                _p3_fig.add_trace(go.Scatter(
                    name=_sup, x=_p3_labels, y=_sup_monthly_price,
                    mode="lines+markers",
                    line=dict(color=_avatar_color(_idx), width=2),
                    marker=dict(size=6),
                    connectgaps=True,
                ))

            _p3_fig.update_layout(
                height=300,
                paper_bgcolor="white", plot_bgcolor="white",
                margin=dict(l=40, r=20, t=20, b=40),
                yaxis=dict(
                    title=f"Avg Price (₹/{uom})",
                    tickformat=",.0f",
                    nticks=15,
                    tickmode="auto",
                    showgrid=True,
                    gridcolor="#e4e9f2",
                    gridwidth=1,
                    griddash="dot",
                    autorange=True,
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.markdown('<div style="height:0.75rem;"></div>', unsafe_allow_html=True)
            st.plotly_chart(_p3_fig, use_container_width=True)

            # Supplier summary table (top 10)
            _p3_display = _p3_totals.head(10)
            _trend_badge = {
                # Procurement perspective: rising prices = bad (red), falling prices = good (green)
                "Rising": '<span class="badge-red">📈 Rising</span>',
                "Falling": '<span class="badge-green">📉 Falling</span>',
                "Stable": '<span class="badge-amber">➡ Stable</span>',
            }
            _p3_rows = ""
            for _, _row in _p3_display.iterrows():
                _p3_rows += f"""
                <tr>
                <td>{_row['entity_name']}</td>
                <td>₹{_row['wtd_price']:,.0f}</td>
                <td>{int(_row['month_count'])}</td>
                <td>{_trend_badge.get(_row['trend'], '')}</td>
                </tr>
                """
            _html(f"""
            <table class="pi-data-table" style="width:100%;border-collapse:collapse;margin-top:0.5rem;">
            <thead><tr>
            <th>Supplier</th><th>Avg Price (₹/{uom})</th><th>Active Months</th><th>Price Trend</th>
            </tr></thead>
            <tbody>{_p3_rows}</tbody>
            </table>
            """)

            # LLM Narrative
            _p3_cache_key = (selected_mol, _f_from_yyyymm, _f_to_yyyymm)
            _p3_cache = st.session_state.get("llm_supplier_price_cache", {})
            if _p3_cache_key in _p3_cache:
                _p3_llm_text = _p3_cache[_p3_cache_key]
            else:
                _p3_top5_summary = [
                    {
                        "supplier": r["entity_name"],
                        "avg_price": round(r["wtd_price"], 2),
                        "trend": r["trend"],
                    }
                    for _, r in _p3_totals.head(5).iterrows()
                ]
                _p3_prompt = (
                    f"You are a pharmaceutical procurement analyst. "
                    f"Analyze the supplier average price trends for {selected_mol} over {month_context}. "
                    f"Top suppliers by avg price: {_p3_top5_summary}. "
                    f"In 3-4 sentences, identify which suppliers have the highest or lowest prices, "
                    f"whether prices are rising or falling, "
                    f"what this implies for Cipla's procurement cost, and give one strategic recommendation "
                    f"for Cipla's sourcing team."
                )
                _p3_llm_text = _llm_analysis(_p3_prompt)
                _p3_cache[_p3_cache_key] = _p3_llm_text
                st.session_state["llm_supplier_price_cache"] = _p3_cache

            if _p3_llm_text:
                _html(f"""
                <div style="background:#f0f9ff;border-left:3px solid #0891b2;border-radius:8px;padding:0.75rem 1rem;margin-top:1rem;font-size:0.85rem;color:#0f172a;line-height:1.6;">
                <span style="font-weight:700;">🤖 AI Analysis · </span>{_p3_llm_text}
                </div>
                """)
            else:
                # Fallback rule-based text
                _p3_rising = [r["entity_name"] for _, r in _p3_totals.iterrows() if r["trend"] == "Rising"]
                _p3_falling = [r["entity_name"] for _, r in _p3_totals.iterrows() if r["trend"] == "Falling"]
                _p3_fallback = f"Supplier avg price analysis for {selected_mol} over {month_context}. "
                if _p3_rising:
                    _p3_fallback += f"Suppliers with rising prices: {', '.join(_p3_rising[:3])}. "
                if _p3_falling:
                    _p3_fallback += f"Suppliers with falling prices: {', '.join(_p3_falling[:3])}. "
                if not _p3_rising and not _p3_falling:
                    _p3_fallback += "All suppliers show stable price patterns. "
                _p3_fallback += "Consider negotiating with higher-priced suppliers or increasing allocation to lower-priced ones."
                _html(f"""
                <div style="background:#f0f9ff;border-left:3px solid #0891b2;border-radius:8px;padding:0.75rem 1rem;margin-top:1rem;font-size:0.85rem;color:#0f172a;line-height:1.6;">
                <span style="font-weight:700;">📊 Analysis · </span>{_p3_fallback}
                </div>
                """)

        _html("</div></div></div>")  # pi-card-content, pi-card, pi-page-body
        _html('<div style="height:1rem;"></div>')

        # ─────────────────────────────────────────────────────────────────────────
        # LLM PANEL 4 — Supplier Volume Shift Analysis
        # ─────────────────────────────────────────────────────────────────────────
        _p4_supplier_df = filtered_df[filtered_df["source"] == "Supplier"]

        _html(f"""
        <div class="pi-page-body">
        <div class="pi-card">
        <div class="pi-section-header">SUPPLIER VOLUME SHIFT ANALYSIS</div>
        <div class="pi-section-title">Supplier Purchase Volume — Increasing &amp; Decreasing Trends</div>
        <div class="pi-section-sub">Supplier perspective · {month_context} · {uom} · MoM volume change</div>
        <div class="pi-card-content" style="margin-top:1rem;">
        """)

        if len(_p4_supplier_df) == 0:
            _html('<div class="pi-info-banner">No supplier data available for the selected filters.</div>')
        else:
            # Step 1 — compute per-supplier monthly volume and MoM % change
            _p4_monthly = (
                _p4_supplier_df.groupby(["entity_name", "yyyymm"])["Sum_of_QTY"]
                .sum()
                .reset_index()
                .sort_values(["entity_name", "yyyymm"])
            )

            _p4_supplier_stats = []
            for _sup_name, _sup_grp in _p4_monthly.groupby("entity_name"):
                _sup_grp = _sup_grp.sort_values("yyyymm")
                _mom_pct = _sup_grp["Sum_of_QTY"].pct_change() * 100
                _avg_mom = _mom_pct.dropna().mean()
                _total_qty = _sup_grp["Sum_of_QTY"].sum()
                if _avg_mom > _VOL_INCREASE_THRESHOLD:
                    _trend = "Increasing"
                elif _avg_mom < _VOL_DECREASE_THRESHOLD:
                    _trend = "Decreasing"
                else:
                    _trend = "Stable"
                _p4_supplier_stats.append({
                    "entity_name": _sup_name,
                    "avg_mom_vol_change": _avg_mom,
                    "total_qty": _total_qty,
                    "trend": _trend,
                })

            _p4_stats_df = pd.DataFrame(_p4_supplier_stats).sort_values("avg_mom_vol_change", ascending=False).reset_index(drop=True)
            _p4_inc_df = _p4_stats_df[_p4_stats_df["trend"] == "Increasing"]
            _p4_dec_df = _p4_stats_df[_p4_stats_df["trend"] == "Decreasing"]

            # Step 2 — two-column layout
            _p4_col_left, _p4_col_right = st.columns(2)

            with _p4_col_left:
                _html('<div style="font-weight:700;color:#16a34a;margin-bottom:0.5rem;">📈 Volume Increasing</div>')
                if len(_p4_inc_df) == 0:
                    _html('<div style="color:#64748b;font-size:0.85rem;">No suppliers with significantly increasing volume.</div>')
                else:
                    _p4_inc_rows = ""
                    for _, _row in _p4_inc_df.iterrows():
                        _p4_inc_rows += f"""
                        <tr>
                        <td>{_row['entity_name']}</td>
                        <td>{fmt_qty(_row['total_qty'])}</td>
                        <td>{_row['avg_mom_vol_change']:+.1f}%</td>
                        <td><span class="badge-green">📈 Increasing</span></td>
                        </tr>
                        """
                    _html(f"""
                    <table class="pi-data-table" style="width:100%;border-collapse:collapse;margin-top:0.5rem;">
                    <thead><tr>
                    <th>Supplier</th><th>Total Vol</th><th>Avg MoM Change</th><th>Status</th>
                    </tr></thead>
                    <tbody>{_p4_inc_rows}</tbody>
                    </table>
                    """)

            with _p4_col_right:
                _html('<div style="font-weight:700;color:#dc2626;margin-bottom:0.5rem;">📉 Volume Decreasing</div>')
                if len(_p4_dec_df) == 0:
                    _html('<div style="color:#64748b;font-size:0.85rem;">No suppliers with significantly decreasing volume.</div>')
                else:
                    _p4_dec_rows = ""
                    for _, _row in _p4_dec_df.iterrows():
                        _p4_dec_rows += f"""
                        <tr>
                        <td>{_row['entity_name']}</td>
                        <td>{fmt_qty(_row['total_qty'])}</td>
                        <td>{_row['avg_mom_vol_change']:+.1f}%</td>
                        <td><span class="badge-red">📉 Decreasing</span></td>
                        </tr>
                        """
                    _html(f"""
                    <table class="pi-data-table" style="width:100%;border-collapse:collapse;margin-top:0.5rem;">
                    <thead><tr>
                    <th>Supplier</th><th>Total Vol</th><th>Avg MoM Change</th><th>Status</th>
                    </tr></thead>
                    <tbody>{_p4_dec_rows}</tbody>
                    </table>
                    """)

            # Step 3 — bar chart (top 10 suppliers by total volume, colored by trend)
            _p4_top10 = _p4_stats_df.nlargest(10, "total_qty")
            _p4_colors = []
            for _t in _p4_top10["trend"]:
                if _t == "Increasing":
                    _p4_colors.append("#16a34a")
                elif _t == "Decreasing":
                    _p4_colors.append("#dc2626")
                else:
                    _p4_colors.append("#d97706")

            _p4_fig = go.Figure(go.Bar(
                x=_p4_top10["total_qty"],
                y=_p4_top10["entity_name"],
                orientation="h",
                marker_color=_p4_colors,
            ))
            # Invisible traces for legend (color coding explanation)
            _p4_fig.add_trace(go.Bar(x=[None], y=[None], orientation="h", name="Increasing", marker_color="#16a34a"))
            _p4_fig.add_trace(go.Bar(x=[None], y=[None], orientation="h", name="Decreasing", marker_color="#dc2626"))
            _p4_fig.add_trace(go.Bar(x=[None], y=[None], orientation="h", name="Stable", marker_color="#d97706"))
            _p4_fig.update_layout(
                height=300,
                paper_bgcolor="white", plot_bgcolor="white",
                margin=dict(l=40, r=20, t=20, b=40),
                xaxis_title=f"Total Volume ({uom})",
                yaxis=dict(autorange="reversed"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                barmode="overlay",
            )
            st.plotly_chart(_p4_fig, use_container_width=True)

            # Step 4 — LLM narrative
            _p4_cache_key = (selected_mol, _f_from_yyyymm, _f_to_yyyymm)
            _p4_cache = st.session_state.get("llm_supplier_vol_shift_cache", {})
            if _p4_cache_key in _p4_cache:
                _p4_llm_text = _p4_cache[_p4_cache_key]
            else:
                _p4_increasing_list = [
                    {"supplier": r["entity_name"], "total_qty": int(r["total_qty"]), "avg_mom_pct": round(r["avg_mom_vol_change"], 1)}
                    for _, r in _p4_inc_df.head(5).iterrows()
                ]
                _p4_decreasing_list = [
                    {"supplier": r["entity_name"], "total_qty": int(r["total_qty"]), "avg_mom_pct": round(r["avg_mom_vol_change"], 1)}
                    for _, r in _p4_dec_df.head(5).iterrows()
                ]
                _p4_prompt = (
                    "You are a pharmaceutical procurement analyst. "
                    f"Analyze the supplier volume shift patterns for {selected_mol} over {month_context}. "
                    f"Suppliers with increasing volumes (avg MoM > {_VOL_INCREASE_THRESHOLD}%): {_p4_increasing_list}. "
                    f"Suppliers with decreasing volumes (avg MoM < {_VOL_DECREASE_THRESHOLD}%): {_p4_decreasing_list}. "
                    f"In 3-4 sentences, explain what these volume shifts mean for supply security, "
                    f"which suppliers are gaining or losing share, "
                    f"and give one procurement risk or opportunity recommendation for Cipla."
                )
                _p4_llm_text = _llm_analysis(_p4_prompt)
                _p4_cache[_p4_cache_key] = _p4_llm_text
                st.session_state["llm_supplier_vol_shift_cache"] = _p4_cache

            if _p4_llm_text:
                _html(f"""
                <div style="background:#f0f9ff;border-left:3px solid #0891b2;border-radius:8px;padding:0.75rem 1rem;margin-top:1rem;font-size:0.85rem;color:#0f172a;line-height:1.6;">
                <span style="font-weight:700;">🤖 AI Analysis · </span>{_p4_llm_text}
                </div>
                """)
            else:
                # Fallback rule-based text
                _p4_fallback = f"Supplier volume shift analysis for {selected_mol} over {month_context}. "
                if len(_p4_inc_df) > 0:
                    _p4_fallback += f"{len(_p4_inc_df)} supplier(s) show significantly increasing volumes. "
                if len(_p4_dec_df) > 0:
                    _p4_fallback += f"{len(_p4_dec_df)} supplier(s) show significantly decreasing volumes. "
                if len(_p4_inc_df) == 0 and len(_p4_dec_df) == 0:
                    _p4_fallback += "All suppliers show stable volume patterns. "
                _p4_fallback += "Monitor suppliers with declining volumes for potential supply risk and consider diversifying to suppliers showing consistent volume growth."
                _html(f"""
                <div style="background:#f0f9ff;border-left:3px solid #0891b2;border-radius:8px;padding:0.75rem 1rem;margin-top:1rem;font-size:0.85rem;color:#0f172a;line-height:1.6;">
                <span style="font-weight:700;">📊 Analysis · </span>{_p4_fallback}
                </div>
                """)

        _html("</div></div></div>")  # pi-card-content, pi-card, pi-page-body
        _html('<div style="height:1rem;"></div>')

        # ─────────────────────────────────────────────────────────────────────────
        # SECTION 2 — Competitor Benchmark (1.5 : 1 layout)
        # ─────────────────────────────────────────────────────────────────────────
        st.markdown('<div class="pi-page-body">', unsafe_allow_html=True)

        # Section 2 uses filtered_df (already filtered by global month selector)
        s2_df = filtered_df

        # Per-entity WTD avg aggregation
        all_ent_agg = (
            s2_df.groupby(["entity_name", "source"])
            .apply(lambda g: pd.Series({
                "wtd_price": _safe_wtd_avg(g["Sum_of_TOTAL_VALUE"], g["Sum_of_QTY"]),
                "total_qty": g["Sum_of_QTY"].sum(),
            }))
            .reset_index()
        )
        all_ent_agg = all_ent_agg[all_ent_agg["wtd_price"] > 0]

        cipla_ent_row = all_ent_agg[all_ent_agg["source"] == "Cipla"]
        non_cipla_rows = all_ent_agg[all_ent_agg["source"] == "Buyer"].sort_values("wtd_price").reset_index(drop=True)
        cipla_bar_price = cipla_ent_row["wtd_price"].mean() if len(cipla_ent_row) > 0 else cipla_price

        # Bar chart data
        s2_market_df = s2_df[s2_df["source"] != "Cipla"]
        s2_market_price = _safe_wtd_avg(s2_market_df["Sum_of_TOTAL_VALUE"], s2_market_df["Sum_of_QTY"])

        bar_items_competitor = []
        for _, row in non_cipla_rows.iterrows():
            bar_items_competitor.append({"entity": row["entity_name"], "price": row["wtd_price"], "type": "competitor", "qty": row["total_qty"]})

        cipla_bar_item = None
        market_bar_item = None
        if len(cipla_ent_row) > 0:
            cipla_bar_item = {"entity": "★ Cipla", "price": cipla_bar_price, "type": "cipla", "qty": cipla_ent_row["total_qty"].sum()}
        if s2_market_price > 0:
            market_bar_item = {"entity": "EXIM Avg", "price": s2_market_price, "type": "market", "qty": s2_market_df["Sum_of_QTY"].sum()}

        def _bar_width(price, min_p, max_p):
            price_range = max_p - min_p if max_p > min_p else 1
            return int(40 + (price - min_p) / price_range * 55)

        s2_left, s2_right = st.columns([3, 2])

        with s2_left:
            # ── Top 25% toggle ────────────────────────────────────────────────────
            bar_view = st.radio(
                "View mode",
                ["Top 25% by Volume", "All Competitors"],
                index=0 if st.session_state["bar_view_mode"] == "Top 25% by Volume" else 1,
                key="bar_view_radio",
                horizontal=True,
                label_visibility="collapsed",
            )
            if bar_view != st.session_state["bar_view_mode"]:
                st.session_state["bar_view_mode"] = bar_view
                st.session_state["bar_page"] = 1
                st.rerun()

            # Determine which competitors to show
            if bar_view == "Top 25% by Volume" and len(non_cipla_rows) > 0:
                volumes = non_cipla_rows["total_qty"].values
                threshold = np.percentile(volumes, 75)
                top25_rows = non_cipla_rows[non_cipla_rows["total_qty"] >= threshold]
                if len(top25_rows) == 0:
                    top25_rows = non_cipla_rows
                display_competitors = top25_rows.reset_index(drop=True)
            else:
                display_competitors = non_cipla_rows

            # Pagination for "All Competitors" mode
            if bar_view == "All Competitors":
                bar_page = st.session_state.get("bar_page", 1)
                total_comp = len(display_competitors)
                page_size = _PAGE_SIZE
                start = (bar_page - 1) * page_size
                end = start + page_size
                page_competitors = display_competitors.iloc[start:end]
            else:
                page_competitors = display_competitors

            # Build the full bar items list for display
            bar_items_display = []
            for _, row in page_competitors.iterrows():
                bar_items_display.append({"entity": row["entity_name"], "price": row["wtd_price"], "type": "competitor", "qty": row["total_qty"]})
            if cipla_bar_item:
                bar_items_display.append(cipla_bar_item)
            if market_bar_item:
                bar_items_display.append(market_bar_item)

            bar_items_sorted = sorted(bar_items_display, key=lambda x: x["price"])

            if bar_items_sorted:
                prices_all = [r["price"] for r in bar_items_sorted]
                min_p = min(prices_all)
                max_p = max(prices_all)
            else:
                min_p = max_p = 0

            # Build HTML bar chart
            bar_html = f"""
            <div class="pi-card" style="margin-bottom:0.5rem;">
              <div class="pi-section-title">WTD Average Price Comparison (₹/{uom})</div>
              <div class="pi-section-sub">Cipla vs competitors · {month_context}</div>
              <div style="padding:0.2rem 0;">
            """
            for r in bar_items_sorted:
                pct = _bar_width(r["price"], min_p, max_p)
                ent_display = r["entity"]
                price_val = r["price"]
                if r["type"] == "cipla":
                    fill = f"background:linear-gradient(90deg,#1d4ed8,#3b82f6);width:{pct}%"
                    lbl_cls = "cipla"
                    badge = '<span class="badge badge-blue">Benchmark</span>'
                elif r["type"] == "market":
                    fill = f"background:rgba(8,145,178,0.35);width:{pct}%"
                    lbl_cls = ""
                    badge = '<span class="badge badge-cyan">EXIM Avg</span>'
                else:
                    vs = ((price_val - cipla_bar_price) / cipla_bar_price * 100) if cipla_bar_price > 0 else 0
                    if vs < 0:
                        fill = f"background:linear-gradient(90deg,#15803d,#16a34a);width:{pct}%"
                        badge = f'<span style="color:#16a34a;font-weight:700;">▼ {abs(vs):.1f}%</span>'
                    elif vs > 15:
                        fill = f"background:linear-gradient(90deg,#b91c1c,#dc2626);width:{pct}%"
                        badge = f'<span style="color:#dc2626;font-weight:700;">▲ {vs:.1f}%</span>'
                    else:
                        fill = f"background:linear-gradient(90deg,#b45309,#d97706);width:{pct}%"
                        badge = f'<span style="color:#d97706;font-weight:700;">▲ {vs:.1f}%</span>'
                    lbl_cls = ""

                bar_html += f"""
                <div class="pi-bar-row">
                  <div class="pi-bar-label {lbl_cls}">{ent_display[:30]}</div>
                  <div class="pi-bar-track">
                    <div class="pi-bar-fill" style="{fill}"></div>
                  </div>
                  <div class="pi-bar-price">₹{price_val:,.0f}</div>
                  <div class="pi-bar-badge">{badge}</div>
                </div>
                """

            if bar_items_sorted:
                mid_p = (min_p + max_p) / 2
                bar_html += f"""
                <div class="pi-bar-scale">
                  <span>₹{min_p:,.0f}</span>
                  <span>₹{mid_p:,.0f}</span>
                  <span>₹{max_p:,.0f}</span>
                </div>
                """
            else:
                bar_html += "<p style='color:var(--t3);'>No data available.</p>"

            bar_html += "</div></div>"
            _html(bar_html)

            # Pagination (only for "All Competitors" mode)
            if bar_view == "All Competitors":
                _pagination_bar(len(display_competitors), _PAGE_SIZE, st.session_state.get("bar_page", 1), "bar_page", "bar")

        with s2_right:
            # Competitor detail table — sort by volume descending, Cipla pinned at top
            non_cipla_by_vol = non_cipla_rows.sort_values("total_qty", ascending=False).reset_index(drop=True)

            # Cipla row first
            table_rows = ""
            if len(cipla_ent_row) > 0:
                cipla_ent_name = cipla_ent_row.iloc[0]["entity_name"]
                cipla_qty_disp = cipla_ent_row["total_qty"].sum()
                table_rows += f"""
                <tr class="cipla-row">
                  <td>
                    <div style="display:flex;align-items:center;gap:7px;">
                      <div class="pi-av" style="background:linear-gradient(135deg,#1d4ed8,#3b82f6);">CI</div>
                      <div>
                        <div style="font-weight:700;color:#1d4ed8;">{cipla_ent_name[:20]}</div>
                        <div style="font-size:0.65rem;color:#64748b;">ERP</div>
                      </div>
                    </div>
                  </td>
                  <td style="font-weight:700;color:#1d4ed8;">₹{cipla_bar_price:,.0f}</td>
                  <td>{fmt_qty(cipla_qty_disp)}</td>
                  <td>—</td>
                  <td><span class="badge badge-blue">Ref</span></td>
                </tr>
                """

            # Paginated non-Cipla rows
            comp_page = st.session_state.get("comp_table_page", 1)
            ct_start = (comp_page - 1) * _PAGE_SIZE
            ct_end = ct_start + _PAGE_SIZE
            page_non_cipla = non_cipla_by_vol.iloc[ct_start:ct_end]

            for idx, (_, row) in enumerate(page_non_cipla.iterrows()):
                ent = row["entity_name"]
                price = row["wtd_price"]
                qty = row["total_qty"]
                vs_pct = ((price - cipla_bar_price) / cipla_bar_price * 100) if cipla_bar_price > 0 else 0
                av_col = _avatar_color(ct_start + idx)
                av_ini = _initials(ent)
                if vs_pct < 0:
                    vs_html = f'<span style="color:#16a34a;font-weight:700;">▼ {abs(vs_pct):.1f}%</span>'
                    pos = '<span class="badge badge-green">Cheaper</span>'
                elif vs_pct > 15:
                    vs_html = f'<span style="color:#dc2626;font-weight:700;">▲ {vs_pct:.1f}%</span>'
                    pos = '<span class="badge badge-red">Premium</span>'
                else:
                    vs_html = f'<span style="color:#d97706;font-weight:700;">▲ {vs_pct:.1f}%</span>'
                    pos = '<span class="badge badge-amber">Higher</span>'
                table_rows += f"""
                <tr>
                  <td>
                    <div style="display:flex;align-items:center;gap:7px;">
                      <div class="pi-av" style="background:{av_col};">{av_ini}</div>
                      <div>
                        <div style="font-weight:500;">{ent[:20]}</div>
                        <div style="font-size:0.65rem;color:#64748b;">EXIM</div>
                      </div>
                    </div>
                  </td>
                  <td>₹{price:,.0f}</td>
                  <td>{fmt_qty(qty)}</td>
                  <td>{vs_html}</td>
                  <td>{pos}</td>
                </tr>
                """

            comp_table = f"""
            <div class="pi-card" style="margin-bottom:0.5rem;">
              <div class="pi-section-title">Price &amp; Volume Summary</div>
              <div class="pi-section-sub">Per entity · WTD average price · {month_context}</div>
              <div style="overflow-x:auto;">
              <table class="pi-comp-table">
                <thead>
                  <tr>
                    <th>Company</th>
                    <th>WTD Avg</th>
                    <th>Volume ({uom})</th>
                    <th>vs Cipla</th>
                    <th>Position</th>
                  </tr>
                </thead>
                <tbody>{table_rows}</tbody>
              </table>
              </div>
            </div>
            """
            _html(comp_table)
            _pagination_bar(len(non_cipla_by_vol), _PAGE_SIZE, comp_page, "comp_table_page", "comp")

        st.markdown("</div>", unsafe_allow_html=True)  # pi-page-body

        # ─────────────────────────────────────────────────────────────────────────
        # SECTION 3 — Bubble Chart (Price over Time)
        # ─────────────────────────────────────────────────────────────────────────
        # Group by entity + yyyymm (buyer-only for competitors, plus Cipla)
        bubble_df = (
            filtered_df[filtered_df["source"].isin(["Buyer", "Cipla"])].groupby(["entity_name", "yyyymm", "source"])
            .apply(lambda g: pd.Series({
                "wtd_price": _safe_wtd_avg(g["Sum_of_TOTAL_VALUE"], g["Sum_of_QTY"]),
                "sum_qty":   g["Sum_of_QTY"].sum(),
                "total_val": g["Sum_of_TOTAL_VALUE"].sum(),
            }))
            .reset_index()
        )
        bubble_df = bubble_df[bubble_df["wtd_price"] > 0].copy()
        bubble_df["month_label"] = bubble_df["yyyymm"].apply(yyyymm_to_label)

        cipla_bubble_ents = set(bubble_df[bubble_df["source"] == "Cipla"]["entity_name"].unique())

        # Top 25% by value: total value per entity across filtered_df
        ent_total_val = bubble_df.groupby("entity_name")["total_val"].sum()
        non_cipla_ents_b = [e for e in ent_total_val.index if e not in cipla_bubble_ents]
        if non_cipla_ents_b:
            val_threshold = np.percentile(ent_total_val[non_cipla_ents_b].values, 75)
            top25_ents_b = set(e for e in non_cipla_ents_b if ent_total_val[e] >= val_threshold)
        else:
            top25_ents_b = set()
        # Always include Cipla entities
        entities_bubble = sorted(top25_ents_b | cipla_bubble_ents)

        # Color map
        entity_colors_b = {}
        b_color_idx = 0
        for ent in sorted(bubble_df["entity_name"].unique()):
            if ent in cipla_bubble_ents:
                entity_colors_b[ent] = "#1d4ed8"
            else:
                entity_colors_b[ent] = _avatar_color(b_color_idx)
                b_color_idx += 1

        # Ordered month labels
        all_month_labels = (
            bubble_df[["yyyymm", "month_label"]]
            .drop_duplicates()
            .sort_values("yyyymm")["month_label"]
            .tolist()
        )

        fig_bubble = go.Figure()

        # Quarterly background bands
        _qb = [
            ("Q1 FY25", "Apr 2024", "Jun 2024", "rgba(59,130,246,0.06)"),
            ("Q2 FY25", "Jul 2024", "Sep 2024", "rgba(16,163,74,0.06)"),
            ("Q3 FY25", "Oct 2024", "Dec 2024", "rgba(217,119,6,0.06)"),
            ("Q4 FY25", "Jan 2025", "Mar 2025", "rgba(220,38,38,0.06)"),
            ("Q1 FY26", "Apr 2025", "Jun 2025", "rgba(59,130,246,0.06)"),
            ("Q2 FY26", "Jul 2025", "Sep 2025", "rgba(16,163,74,0.06)"),
            ("Q3 FY26", "Oct 2025", "Dec 2025", "rgba(217,119,6,0.06)"),
            ("Q4 FY26", "Jan 2026", "Mar 2026", "rgba(220,38,38,0.06)"),
        ]
        for q_lbl, q_s, q_e, q_col in _qb:
            s_in = q_s in all_month_labels
            e_in = q_e in all_month_labels
            if s_in or e_in:
                x0 = q_s if s_in else all_month_labels[0]
                x1 = q_e if e_in else all_month_labels[-1]
                fig_bubble.add_vrect(
                    x0=x0, x1=x1,
                    fillcolor=q_col, opacity=1, layer="below", line_width=0,
                    annotation_text=q_lbl, annotation_position="top left",
                    annotation_font_size=9, annotation_font_color="#64748b",
                )

        # Build Cipla monthly lookup for comparison in hover tooltips
        cipla_monthly_lookup = {}
        for _, row in bubble_df[bubble_df["source"] == "Cipla"].iterrows():
            cipla_monthly_lookup[row["yyyymm"]] = {
                "wtd_price": row["wtd_price"],
                "sum_qty": row["sum_qty"],
            }

        cipla_mol_label = f"cipla-{selected_mol.lower()}"

        for ent in entities_bubble:
            ent_df = bubble_df[bubble_df["entity_name"] == ent].sort_values("yyyymm")
            color = entity_colors_b.get(ent, "#64748b")
            # Cipla traces use "cipla-<molecule>" as the display name
            is_cipla = ent in cipla_bubble_ents
            trace_name = cipla_mol_label if is_cipla else ent

            if is_cipla:
                custom = list(zip(
                    ent_df["sum_qty"],
                    ent_df["yyyymm"],
                ))
                htemplate = (
                    f"<b>{trace_name}</b> <i>(Benchmark)</i><br>"
                    "Month: %{x}<br>"
                    f"Avg Price: ₹%{{y:,.0f}} /{uom}<br>"
                    f"Volume: %{{customdata[0]:,.0f}} {uom}<extra></extra>"
                )
            else:
                rows_custom = []
                for _, r in ent_df.iterrows():
                    c_info = cipla_monthly_lookup.get(r["yyyymm"], {})
                    c_price = c_info.get("wtd_price", 0)
                    c_qty = c_info.get("sum_qty", 0)
                    p_diff = r["wtd_price"] - c_price if c_price else float("nan")
                    p_pct = (p_diff / c_price * 100) if c_price else float("nan")
                    rows_custom.append((
                        r["sum_qty"],
                        r["yyyymm"],
                        c_price,
                        c_qty,
                        p_diff,
                        p_pct,
                    ))
                custom = rows_custom
                htemplate = (
                    f"<b>{trace_name}</b><br>"
                    "Month: %{x}<br>"
                    f"Avg Price: ₹%{{y:,.0f}} /{uom}<br>"
                    f"Volume: %{{customdata[0]:,.0f}} {uom}<br>"
                    "─────────────────────────────<br>"
                    f"<b>vs {cipla_mol_label}</b><br>"
                    f"  Price:  ₹%{{y:,.0f}}  vs  ₹%{{customdata[2]:,.0f}}  →  %{{customdata[4]:+,.0f}} (%{{customdata[5]:+.1f}}%)<br>"
                    f"  Volume: %{{customdata[0]:,.0f}}  vs  %{{customdata[3]:,.0f}} {uom}<extra></extra>"
                )

            fig_bubble.add_trace(go.Scatter(
                x=ent_df["month_label"],
                y=ent_df["wtd_price"],
                mode="markers",
                name=trace_name,
                marker=dict(
                    size=10,
                    color=color,
                    opacity=0.75,
                    line=dict(width=2, color="white"),
                ),
                customdata=custom,
                hovertemplate=htemplate,
            ))

        # Cipla reference line
        if cipla_price > 0 and all_month_labels:
            fig_bubble.add_hline(
                y=cipla_price,
                line_dash="dash", line_color="#1d4ed8", line_width=1.5,
                annotation_text=f"Cipla ₹{cipla_price:,.0f}",
                annotation_position="right",
                annotation_font_size=10, annotation_font_color="#1d4ed8",
            )

        fig_bubble.update_layout(
            height=420,
            paper_bgcolor="white",
            plot_bgcolor="#fafbff",
            xaxis_title="Month",
            yaxis_title=f"Avg Price (₹/{uom})",
            xaxis=dict(
                categoryorder="array",
                categoryarray=all_month_labels,
                showgrid=True, gridcolor="#e4e9f2", gridwidth=1, griddash="dot",
            ),
            yaxis=dict(showgrid=True, gridcolor="#e4e9f2", gridwidth=1, griddash="dot"),
            legend=dict(
                orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
                font=dict(size=10),
            ),
            margin=dict(l=10, r=130, t=30, b=10),
            font=dict(size=11),
        )

        st.markdown('<div class="pi-page-body">', unsafe_allow_html=True)
        _html(f"""
        <div class="pi-card" style="margin-bottom:0.5rem;">
          <div class="pi-section-title">Price over Time · Dot Chart</div>
          <div class="pi-section-sub">X = Month · Y = Avg Price (₹/{uom}) · Dot = Entity · Hover for Cipla comparison</div>
        </div>
        """)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="pi-page-body">', unsafe_allow_html=True)
        st.plotly_chart(fig_bubble, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ─────────────────────────────────────────────────────────────────────────
        # SECTION 4 — Volume & Price Data Tables
        # ─────────────────────────────────────────────────────────────────────────
        st.markdown('<div class="pi-page-body">', unsafe_allow_html=True)
        t4_l, t4_r = st.columns(2)

        with t4_l:
            # Cipla Monthly table — driven by main top-level filters
            cipla_filtered_full = filtered_df[filtered_df["source"] == "Cipla"]

            cipla_sorted = cipla_filtered_full.sort_values("yyyymm").reset_index(drop=True)

            # Pagination
            cipla_pg = st.session_state.get("cipla_table_page", 1)
            cip_start = (cipla_pg - 1) * _PAGE_SIZE
            cip_end = cip_start + _PAGE_SIZE
            cipla_monthly_page = cipla_sorted.iloc[cip_start:cip_end]

            max_qty_idx = cipla_monthly_page["Sum_of_QTY"].idxmax() if len(cipla_monthly_page) > 0 else None
            min_qty_idx = cipla_monthly_page["Sum_of_QTY"].idxmin() if len(cipla_monthly_page) > 1 else None

            cip_rows_html = ""
            for idx_r, row_r in cipla_monthly_page.iterrows():
                qty_style = ""
                if max_qty_idx is not None and idx_r == max_qty_idx:
                    qty_style = 'style="color:#16a34a;font-weight:700;"'
                elif min_qty_idx is not None and idx_r == min_qty_idx and len(cipla_monthly_page) > 1:
                    qty_style = 'style="color:#dc2626;font-weight:700;"'
                cip_rows_html += f"""
                <tr>
                  <td>{yyyymm_to_label(row_r["yyyymm"])}</td>
                  <td>{row_r["uom"]}</td>
                  <td>{row_r["GRADE_SPEC"]}</td>
                  <td {qty_style}>{fmt_qty(row_r["Sum_of_QTY"])}</td>
                  <td>{fmt_inr(row_r["Sum_of_TOTAL_VALUE"])}</td>
                  <td style="color:#1d4ed8;font-weight:700;">₹{row_r["Avg_PRICE"]:,.0f}</td>
                </tr>
                """

            total_qty_c = cipla_filtered_full["Sum_of_QTY"].sum()
            total_val_c = cipla_filtered_full["Sum_of_TOTAL_VALUE"].sum()
            wtd_avg_c = _safe_wtd_avg(cipla_filtered_full["Sum_of_TOTAL_VALUE"], cipla_filtered_full["Sum_of_QTY"])

            cip_rows_html += f"""
            <tr class="footer-row">
              <td>WTD Avg</td>
              <td>—</td>
              <td>—</td>
              <td>{fmt_qty(total_qty_c)}</td>
              <td>{fmt_inr(total_val_c)}</td>
              <td style="color:#1d4ed8;font-weight:700;">₹{wtd_avg_c:,.0f}</td>
            </tr>
            """

            _html(f"""
    <div class="pi-card" style="margin-bottom:0.5rem;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem;">
    <div>
    <div class="pi-section-title">Cipla — Monthly Price &amp; Volume</div>
    <div class="pi-section-sub">api = {selected_mol.upper()} · {month_context} · per grade &amp; UOM · Sum QTY · Avg PRICE</div>
    </div>
    <span class="badge badge-blue">ERP</span>
    </div>
    <div style="overflow-x:auto;">
    <table class="pi-data-table">
    <thead>
    <tr>
    <th>PERIOD</th><th>UOM</th><th>GRADE</th>
    <th>SUM OF QTY</th><th>TOTAL VALUE</th><th>AVG PRICE</th>
    </tr>
    </thead>
    <tbody>{cip_rows_html}</tbody>
    </table>
    </div>
    </div>
    """)
            _pagination_bar(len(cipla_sorted), _PAGE_SIZE, cipla_pg, "cipla_table_page", "cipla")

        with t4_r:
            # EXIM Supplier table — sort by qty descending, supplier-only rows
            supplier_df_f = filtered_df[filtered_df["source"] == "Supplier"]
            exim_agg = (
                supplier_df_f.groupby("entity_name")
                .apply(lambda g: pd.Series({
                    "grade_e":   g["GRADE_SPEC"].mode()[0] if len(g) > 0 else grade,
                    "uom_e":     g["uom"].mode()[0] if len(g) > 0 else uom,
                    "sum_qty":   g["Sum_of_QTY"].sum(),
                    "total_val": g["Sum_of_TOTAL_VALUE"].sum(),
                    "avg_price": _safe_wtd_avg(g["Sum_of_TOTAL_VALUE"], g["Sum_of_QTY"]),
                }))
                .reset_index()
                .sort_values("sum_qty", ascending=False)
                .reset_index(drop=True)
            )

            # Pagination
            exim_pg = st.session_state.get("exim_table_page", 1)
            ex_start = (exim_pg - 1) * _PAGE_SIZE
            ex_end = ex_start + _PAGE_SIZE
            exim_page = exim_agg.iloc[ex_start:ex_end]

            exim_rows_html = ""
            for _, row_e in exim_page.iterrows():
                vs_diff = row_e["avg_price"] - cipla_price
                if cipla_price > 0:
                    if vs_diff < 0:
                        vs_html = f'<span style="color:#16a34a;font-weight:700;">▼ ₹{abs(vs_diff):,.0f}</span>'
                        price_style = 'style="color:#16a34a;font-weight:700;"'
                    else:
                        vs_html = f'<span style="color:#dc2626;font-weight:700;">▲ ₹{vs_diff:,.0f}</span>'
                        price_style = 'style="color:#dc2626;font-weight:700;"'
                else:
                    vs_html = "—"
                    price_style = ""
                exim_rows_html += f"""
                <tr>
                  <td style="font-weight:500;">{row_e["entity_name"][:26]}</td>
                  <td>{row_e["grade_e"]}</td>
                  <td>{fmt_qty(row_e["sum_qty"])}</td>
                  <td>{fmt_inr(row_e["total_val"])}</td>
                  <td {price_style}>₹{row_e["avg_price"]:,.0f}</td>
                  <td>{vs_html}</td>
                </tr>
                """

            total_qty_e = supplier_df_f["Sum_of_QTY"].sum()
            total_val_e = supplier_df_f["Sum_of_TOTAL_VALUE"].sum()
            wtd_avg_e = _safe_wtd_avg(supplier_df_f["Sum_of_TOTAL_VALUE"], supplier_df_f["Sum_of_QTY"])
            vs_footer_diff = wtd_avg_e - cipla_price
            if cipla_price > 0:
                vs_footer = (
                    f'<span style="color:#16a34a;font-weight:700;">▼ ₹{abs(vs_footer_diff):,.0f}</span>'
                    if vs_footer_diff < 0
                    else f'<span style="color:#dc2626;font-weight:700;">▲ ₹{vs_footer_diff:,.0f}</span>'
                )
            else:
                vs_footer = "—"

            exim_rows_html += f"""
            <tr class="footer-row">
              <td>EXIM WTD Avg</td>
              <td>—</td>
              <td>{fmt_qty(total_qty_e)}</td>
              <td>{fmt_inr(total_val_e)}</td>
              <td>₹{wtd_avg_e:,.0f}</td>
              <td>{vs_footer}</td>
            </tr>
            """

            _html(f"""
    <div class="pi-card" style="margin-bottom:0.5rem;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem;">
    <div>
    <div class="pi-section-title">EXIM — Supplier Price &amp; Volume</div>
    <div class="pi-section-sub">supplier · {month_context} · {uom} · Grade · Sum QTY · Avg PRICE</div>
    </div>
    <span class="badge badge-cyan">EXIM</span>
    </div>
    <div style="overflow-x:auto;">
    <table class="pi-data-table">
    <thead>
    <tr>
    <th>SUPPLIER</th><th>GRADE</th>
    <th>SUM OF QTY</th><th>TOTAL VALUE</th><th>AVG PRICE</th><th>VS CIPLA</th>
    </tr>
    </thead>
    <tbody>{exim_rows_html}</tbody>
    </table>
    </div>
    </div>
    """)
            _pagination_bar(len(exim_agg), _PAGE_SIZE, exim_pg, "exim_table_page", "exim")

        st.markdown("</div>", unsafe_allow_html=True)  # pi-page-body

        # ─────────────────────────────────────────────────────────────────────────
        # FOOTER
        # ─────────────────────────────────────────────────────────────────────────
        st.markdown("""
        <div class="pi-footer">
          <div>
            <span style="font-size:1.2rem;margin-right:0.5rem;">💊</span>
            PharmaIntel | Price Benchmarking Intelligence Platform |
            Data Sources: Internal ERP · EXIM Trade Data |
            Confidential — For Internal Use Only
          </div>
          <div>
            <a href="#">Documentation</a>
            <a href="#">Export</a>
            <a href="#">Support</a>
            <a href="#">© 2026 Cipla Ltd</a>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ─── LANDING PAGE (no molecule selected) ─────────────────────────────────────
else:
    st.markdown("""
    <div style="text-align:center;padding:3rem 2rem 2rem 2rem;">
      <div style="font-size:3rem;margin-bottom:1rem;">💊</div>
      <h2 style="color:#0f172a;font-size:1.8rem;font-weight:800;margin-bottom:0.5rem;">
        Welcome to PharmaIntel
      </h2>
      <p style="color:#64748b;font-size:1rem;max-width:600px;margin:0 auto 0 auto;">
        Search for a molecule above and click <strong>Analyse</strong> to explore
        Cipla's procurement intelligence — price benchmarks, EXIM competitor analysis,
        and monthly trend data.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="pi-footer" style="margin-top:3rem;">
      <div>
        <span style="font-size:1.2rem;margin-right:0.5rem;">💊</span>
        PharmaIntel | Price Benchmarking Intelligence Platform |
        Data Sources: Internal ERP · EXIM Trade Data |
        Confidential — For Internal Use Only
      </div>
      <div>
        <a href="#">Documentation</a>
        <a href="#">Export</a>
        <a href="#">Support</a>
        <a href="#">© 2026 Cipla Ltd</a>
      </div>
    </div>
    """, unsafe_allow_html=True)
