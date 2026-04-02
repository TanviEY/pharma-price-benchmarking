# streamlit_app.py
import calendar
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

from src.file_discovery import FileDiscovery
from src.fuzzy_matcher import FuzzyMatcher
from src.pipeline import run_processing_pipeline
from src.utils import Utils
from src.settings import MOLECULE_MAPPING


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


PERIOD_OPTIONS = {
    "FY 2025–26": ("202504", "202603"),
    "FY 2024–25": ("202404", "202503"),
    "Q4 FY26":    ("202601", "202603"),
    "Q3 FY26":    ("202610", "202612"),
    "Q2 FY26":    ("202607", "202609"),
    "Q1 FY26":    ("202604", "202606"),
    "All Time":   (None, None),
}


def filter_by_period(df, period_label):
    start, end = PERIOD_OPTIONS.get(period_label, (None, None))
    if start is None:
        return df
    return df[(df["yyyymm"] >= start) & (df["yyyymm"] <= end)]


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
.pi-section-hdr {
  font-size:0.7rem; font-weight:700; color:var(--t3);
  text-transform:uppercase; letter-spacing:0.8px; margin-bottom:0.2rem;
}
.pi-section-title { font-size:1rem; font-weight:700; color:var(--t1); margin-bottom:0.2rem; }
.pi-section-sub { font-size:0.78rem; color:var(--t3); margin-bottom:0.8rem; }

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
</style>
""", unsafe_allow_html=True)

# ─── session state ────────────────────────────────────────────────────────────
for _k, _v in [
    ("selected_molecule", None),
    ("selected_period", "FY 2025–26"),
    ("pipeline_result", None),
    ("chart_month_filter", "All Months"),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _on_mol_enter():
    st.session_state["_analyse_trigger"] = True

# ─── init objects ─────────────────────────────────────────────────────────────
file_discovery = FileDiscovery(data_dir="data/raw", molecule_mapping=MOLECULE_MAPPING)
fuzzy_matcher = FuzzyMatcher(molecule_mapping=MOLECULE_MAPPING, threshold=70)
available_molecules = file_discovery.get_available_molecules()

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
    sc1, sc2, sc3 = st.columns([4, 2, 2])
    with sc1:
        hero_mol_input = st.text_input(
            "Molecule",
            value=st.session_state.selected_molecule or "",
            placeholder="e.g., Azithromycin…",
            key="hero_mol_input",
            on_change=_on_mol_enter,
        )
    with sc2:
        period_keys = list(PERIOD_OPTIONS.keys())
        default_idx = period_keys.index(st.session_state.selected_period) if st.session_state.selected_period in period_keys else 0
        hero_period = st.selectbox(
            "Period",
            period_keys,
            index=default_idx,
            key="hero_period_sel",
        )
    with sc3:
        hero_origin = st.selectbox(
            "Origin",
            ["All Origins", "India", "China", "EU / US"],
            key="hero_origin_sel",
        )
    st.markdown('<div class="pi-analyse-btn">', unsafe_allow_html=True)
    analyse_clicked = st.button("Analyse", key="hero_analyse_btn")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Check Enter-key trigger
_enter_triggered = st.session_state.pop("_analyse_trigger", False)

# Suggestion panel (shown when there is typed input and no molecule is currently loaded)
if hero_mol_input.strip() and not st.session_state.selected_molecule:
    suggestions = fuzzy_matcher.get_suggestions(hero_mol_input.strip(), top_n=5)
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
                if st.button(mol_name.upper(), key=f"sug_{mol_name}_{i}"):
                    st.session_state.selected_molecule = mol_name
                    st.session_state.selected_period = hero_period
                    st.session_state.pipeline_result = None
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# Handle Analyse click or Enter key
if (analyse_clicked or _enter_triggered) and hero_mol_input.strip():
    top_match = fuzzy_matcher.get_top_match(hero_mol_input.strip())
    if top_match and top_match in available_molecules:
        st.session_state.selected_molecule = top_match
        st.session_state.selected_period = hero_period
        st.session_state.pipeline_result = None
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
    selected_period = st.session_state.selected_period

    # ── run pipeline ──────────────────────────────────────────────────────────
    with st.spinner(f"Loading data for {selected_mol.upper()}…"):
        result = run_processing_pipeline(selected_mol, file_discovery)

    if result["status"] == "failed":
        st.markdown(
            f'<div class="pi-info-banner">❌ Pipeline failed: '
            f'{", ".join(result["errors"])}</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    consolidated_df = result["data"]["consolidated"]

    # Apply period filter
    filtered_df = filter_by_period(consolidated_df, selected_period)
    if len(filtered_df) == 0:
        filtered_df = consolidated_df.copy()
        st.markdown(
            f'<div class="pi-info-banner" style="margin:1rem 1.5rem 0 1.5rem;">'
            f'ℹ️ No data found for <strong>{selected_period}</strong>. '
            f'Showing all available data.</div>',
            unsafe_allow_html=True,
        )

    # Metadata
    mol_cfg = MOLECULE_MAPPING["molecules"].get(selected_mol, {})
    cas_code = mol_cfg.get("cipla_api_filter", selected_mol.upper())
    uom = filtered_df["uom"].mode()[0] if len(filtered_df) > 0 else "KG"
    grade_series = filtered_df[filtered_df["source"] == "Cipla"]["GRADE_SPEC"]
    grade = grade_series.mode()[0] if len(grade_series) > 0 else "USP"
    period_label = selected_period

    # Build month filter options for Section 2
    available_months_raw = sorted(filtered_df["yyyymm"].unique())
    available_months_labels = ["All Months"] + [yyyymm_to_label(m) for m in available_months_raw]
    month_label_to_yyyymm = {yyyymm_to_label(m): m for m in available_months_raw}

    # ── MATERIAL BANNER ──────────────────────────────────────────────────────
    export_csv = filtered_df.to_csv(index=False).encode("utf-8")

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
                <span class="pi-chip">{period_label}</span>
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
            file_name=f"{selected_mol}_{selected_period.replace(' ', '_').replace('–', '-')}.csv",
            mime="text/csv",
            key="excel_export",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1 — 5 KPI Cards with Sparklines
    # ─────────────────────────────────────────────────────────────────────────
    cipla_df_f = filtered_df[filtered_df["source"] == "Cipla"]
    market_df_f = filtered_df[filtered_df["source"] != "Cipla"]

    cipla_price = _safe_wtd_avg(cipla_df_f["Sum_of_TOTAL_VALUE"], cipla_df_f["Sum_of_QTY"])
    market_price = _safe_wtd_avg(market_df_f["Sum_of_TOTAL_VALUE"], market_df_f["Sum_of_QTY"])
    cipla_n_records = len(cipla_df_f)
    cipla_total_qty = cipla_df_f["Sum_of_QTY"].sum()
    market_n_ent = market_df_f["entity_name"].nunique()

    # Per-entity WTD avg
    low_price = high_price = 0.0
    low_ent = high_ent = "—"
    if len(market_df_f) > 0:
        ent_prices = (
            market_df_f.groupby("entity_name")
            .apply(lambda g: _safe_wtd_avg(g["Sum_of_TOTAL_VALUE"], g["Sum_of_QTY"]))
            .reset_index(name="wtd_price")
        )
        ent_prices = ent_prices[ent_prices["wtd_price"] > 0]
        if len(ent_prices) > 0:
            min_row = ent_prices.loc[ent_prices["wtd_price"].idxmin()]
            max_row = ent_prices.loc[ent_prices["wtd_price"].idxmax()]
            low_price, low_ent = min_row["wtd_price"], min_row["entity_name"]
            high_price, high_ent = max_row["wtd_price"], max_row["entity_name"]

    cost_adv = cipla_price - market_price if market_price > 0 else 0.0
    cost_pct = abs(cost_adv / market_price * 100) if market_price > 0 else 0.0

    # Sparklines — monthly WTD avg
    months_sorted = sorted(filtered_df["yyyymm"].unique())

    def _monthly_wtd(src_fn, months):
        vals = []
        for m in months:
            mdf = filtered_df[filtered_df["yyyymm"] == m]
            mdf = src_fn(mdf)
            vals.append(_safe_wtd_avg(mdf["Sum_of_TOTAL_VALUE"], mdf["Sum_of_QTY"]))
        return vals

    cipla_spark = _render_sparkline(
        _monthly_wtd(lambda d: d[d["source"] == "Cipla"], months_sorted), "#3b82f6"
    )
    market_spark = _render_sparkline(
        _monthly_wtd(lambda d: d[d["source"] != "Cipla"], months_sorted), "#0891b2"
    )

    # Cost advantage badge
    if cipla_price > 0 and market_price > 0:
        adv_sym = "▼" if cost_adv < 0 else "▲"
        adv_word = "below" if cost_adv < 0 else "above"
        adv_text = f"{adv_sym} {cost_pct:.1f}% {adv_word} market"
    else:
        adv_text = period_label

    st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="pi-page-body">', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.markdown(f"""
        <div class="pi-kpi-card" style="border-top-color:#3b82f6;">
          <div class="pi-kpi-label">Cipla WTD Avg · ERP</div>
          <div class="pi-kpi-value">₹{cipla_price:,.0f} <span>/{uom}</span></div>
          <div><span class="pi-kpi-badge" style="background:#eff6ff;color:#1d4ed8;">{period_label}</span></div>
          <div class="pi-kpi-note">{cipla_n_records} POs · {fmt_qty(cipla_total_qty)} {uom}</div>
          {cipla_spark}
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="pi-kpi-card" style="border-top-color:#0891b2;">
          <div class="pi-kpi-label">EXIM Market Avg</div>
          <div class="pi-kpi-value">₹{market_price:,.0f} <span>/{uom}</span></div>
          <div><span class="pi-kpi-badge" style="background:#ecfeff;color:#0891b2;">{market_n_ent} competitors</span></div>
          <div class="pi-kpi-note">EXIM data</div>
          {market_spark}
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
          <div class="pi-kpi-note">period avg</div>
        </div>
        """, unsafe_allow_html=True)

    with k5:
        st.markdown(f"""
        <div class="pi-kpi-card" style="border-top-color:#dc2626;">
          <div class="pi-kpi-label">Highest Competitor</div>
          <div class="pi-kpi-value">₹{high_price:,.0f} <span>/{uom}</span></div>
          <div><span class="pi-kpi-badge" style="background:#fff1f2;color:#dc2626;">{high_ent[:22] if high_ent != "—" else "—"}</span></div>
          <div class="pi-kpi-note">premium grade</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # pi-page-body
    st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2 — Competitor Benchmark (1.5 : 1 layout)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="pi-page-body">', unsafe_allow_html=True)

    # Month filter row — sits cleanly above the two Section 2 cards
    filt_col_l, filt_col_r = st.columns([5, 2])
    with filt_col_l:
        st.markdown(
            '<p style="font-size:0.78rem;color:#64748b;margin:0.5rem 0 0.2rem 0;">'
            'Benchmark Comparison · Filter by month to drill down</p>',
            unsafe_allow_html=True
        )
    with filt_col_r:
        st.markdown('<div class="pi-month-filter">', unsafe_allow_html=True)
        _saved_month = st.session_state.get("chart_month_filter", "All Months")
        chart_month = st.selectbox(
            "Filter month",
            available_months_labels,
            index=available_months_labels.index(_saved_month) if _saved_month in available_months_labels else 0,
            key="chart_month_sel",
            label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)
        st.session_state["chart_month_filter"] = chart_month

    # Apply month filter for Section 2 charts only
    if chart_month != "All Months" and chart_month in month_label_to_yyyymm:
        s2_df = filtered_df[filtered_df["yyyymm"] == month_label_to_yyyymm[chart_month]]
    else:
        s2_df = filtered_df

    month_context = chart_month if chart_month != "All Months" else period_label

    # Per-entity WTD avg aggregation (uses s2_df for Section 2 only)
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
    non_cipla_rows = all_ent_agg[all_ent_agg["source"] != "Cipla"].sort_values("wtd_price").reset_index(drop=True)
    cipla_bar_price = cipla_ent_row["wtd_price"].mean() if len(cipla_ent_row) > 0 else cipla_price

    # Bar chart data
    s2_market_df = s2_df[s2_df["source"] != "Cipla"]
    s2_market_price = _safe_wtd_avg(s2_market_df["Sum_of_TOTAL_VALUE"], s2_market_df["Sum_of_QTY"])

    bar_items = []
    for _, row in non_cipla_rows.iterrows():
        bar_items.append({"entity": row["entity_name"], "price": row["wtd_price"], "type": "competitor", "qty": row["total_qty"]})
    if len(cipla_ent_row) > 0:
        bar_items.append({"entity": "★ Cipla", "price": cipla_bar_price, "type": "cipla", "qty": cipla_ent_row["total_qty"].sum()})
    if s2_market_price > 0:
        bar_items.append({"entity": "EXIM Avg", "price": s2_market_price, "type": "market", "qty": s2_market_df["Sum_of_QTY"].sum()})

    bar_items_sorted = sorted(bar_items, key=lambda x: x["price"])

    def _bar_width(price, min_p, max_p):
        price_range = max_p - min_p if max_p > min_p else 1
        return int(40 + (price - min_p) / price_range * 55)

    if bar_items_sorted:
        prices_all = [r["price"] for r in bar_items_sorted]
        min_p = min(prices_all)
        max_p = max(prices_all)
    else:
        min_p = max_p = 0

    s2_left, s2_right = st.columns([3, 2])

    with s2_left:
        # Build HTML bar chart
        bar_html = f"""
        <div class="pi-card" style="margin-bottom:1.5rem;">
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

    with s2_right:
        # Competitor detail table
        table_rows = ""

        # Cipla row first
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

        for idx, (_, row) in enumerate(non_cipla_rows.iterrows()):
            ent = row["entity_name"]
            price = row["wtd_price"]
            qty = row["total_qty"]
            vs_pct = ((price - cipla_bar_price) / cipla_bar_price * 100) if cipla_bar_price > 0 else 0
            av_col = _avatar_color(idx)
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
        <div class="pi-card" style="margin-bottom:1.5rem;">
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

    st.markdown("</div>", unsafe_allow_html=True)  # pi-page-body

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3 — Bubble Chart (Price over Time)
    # ─────────────────────────────────────────────────────────────────────────
    # Group by entity + yyyymm
    bubble_df = (
        filtered_df.groupby(["entity_name", "yyyymm", "source"])
        .apply(lambda g: pd.Series({
            "wtd_price": _safe_wtd_avg(g["Sum_of_TOTAL_VALUE"], g["Sum_of_QTY"]),
            "sum_qty":   g["Sum_of_QTY"].sum(),
        }))
        .reset_index()
    )
    bubble_df = bubble_df[bubble_df["wtd_price"] > 0].copy()
    bubble_df["month_label"] = bubble_df["yyyymm"].apply(yyyymm_to_label)

    entities_bubble = sorted(bubble_df["entity_name"].unique())
    cipla_bubble_ents = set(bubble_df[bubble_df["source"] == "Cipla"]["entity_name"].unique())

    # Scale bubble sizes: sqrt(qty / max_qty) * 60 + 10
    max_qty_b = bubble_df["sum_qty"].max() if bubble_df["sum_qty"].max() > 0 else 1
    bubble_df["bubble_size"] = ((bubble_df["sum_qty"] / max_qty_b) ** 0.5) * 60 + 10

    # Color map
    entity_colors_b = {}
    b_color_idx = 0
    for ent in entities_bubble:
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

    for ent in entities_bubble:
        ent_df = bubble_df[bubble_df["entity_name"] == ent].sort_values("yyyymm")
        color = entity_colors_b.get(ent, "#64748b")
        fig_bubble.add_trace(go.Scatter(
            x=ent_df["month_label"],
            y=ent_df["wtd_price"],
            mode="markers",
            name=ent,
            marker=dict(
                size=ent_df["bubble_size"].tolist(),
                sizemode="diameter",
                sizeref=1,
                color=color,
                opacity=0.75,
                line=dict(width=2, color="white"),
            ),
            customdata=list(zip(ent_df["sum_qty"], ent_df["yyyymm"])),
            hovertemplate=(
                f"<b>{ent}</b><br>"
                "Month: %{x}<br>"
                f"Avg Price: ₹%{{y:,.0f}} /{uom}<br>"
                f"Volume: %{{customdata[0]:,.0f}} {uom}<extra></extra>"
            ),
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
      <div class="pi-section-title">Price over Time · Bubble Analysis</div>
      <div class="pi-section-sub">X = Month · Y = Avg Price (₹/{uom}) · Bubble size = Volume · Colour = Entity</div>
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
        # Cipla Monthly table
        cipla_monthly = (
            cipla_df_f.groupby("yyyymm")
            .apply(lambda g: pd.Series({
                "uom_m":     g["uom"].mode()[0] if len(g) > 0 else uom,
                "grade_m":   g["GRADE_SPEC"].mode()[0] if len(g) > 0 else grade,
                "sum_qty":   g["Sum_of_QTY"].sum(),
                "total_val": g["Sum_of_TOTAL_VALUE"].sum(),
                "avg_price": _safe_wtd_avg(g["Sum_of_TOTAL_VALUE"], g["Sum_of_QTY"]),
            }))
            .reset_index()
            .sort_values("yyyymm")
        )

        max_qty_idx = cipla_monthly["sum_qty"].idxmax() if len(cipla_monthly) > 0 else None
        min_qty_idx = cipla_monthly["sum_qty"].idxmin() if len(cipla_monthly) > 1 else None

        cip_rows_html = ""
        for idx_r, row_r in cipla_monthly.iterrows():
            qty_style = ""
            if max_qty_idx is not None and idx_r == max_qty_idx:
                qty_style = 'style="color:#16a34a;font-weight:700;"'
            elif min_qty_idx is not None and idx_r == min_qty_idx and len(cipla_monthly) > 1:
                qty_style = 'style="color:#dc2626;font-weight:700;"'
            cip_rows_html += f"""
            <tr>
              <td>{yyyymm_to_label(row_r["yyyymm"])}</td>
              <td>{row_r["uom_m"]}</td>
              <td>{row_r["grade_m"]}</td>
              <td {qty_style}>{fmt_qty(row_r["sum_qty"])}</td>
              <td>{fmt_inr(row_r["total_val"])}</td>
              <td style="color:#1d4ed8;font-weight:700;">₹{row_r["avg_price"]:,.0f}</td>
            </tr>
            """

        total_qty_c = cipla_df_f["Sum_of_QTY"].sum()
        total_val_c = cipla_df_f["Sum_of_TOTAL_VALUE"].sum()
        wtd_avg_c = _safe_wtd_avg(cipla_df_f["Sum_of_TOTAL_VALUE"], cipla_df_f["Sum_of_QTY"])
        grade_c = cipla_df_f["GRADE_SPEC"].mode()[0] if len(cipla_df_f) > 0 else grade
        uom_c = cipla_df_f["uom"].mode()[0] if len(cipla_df_f) > 0 else uom

        cip_rows_html += f"""
        <tr class="footer-row">
          <td>WTD Avg</td>
          <td>{uom_c}</td>
          <td>{grade_c}</td>
          <td>{fmt_qty(total_qty_c)}</td>
          <td>{fmt_inr(total_val_c)}</td>
          <td style="color:#1d4ed8;font-weight:700;">₹{wtd_avg_c:,.0f}</td>
        </tr>
        """

        _html(f"""
<div class="pi-card" style="margin-bottom:1.5rem;">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem;">
<div>
<div class="pi-section-title">Cipla — Monthly Price &amp; Volume</div>
<div class="pi-section-sub">api = {selected_mol.upper()} · {period_label} · {uom_c} · {grade_c} · Sum QTY · Avg PRICE</div>
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

    with t4_r:
        # EXIM Supplier table
        exim_agg = (
            market_df_f.groupby("entity_name")
            .apply(lambda g: pd.Series({
                "grade_e":   g["GRADE_SPEC"].mode()[0] if len(g) > 0 else grade,
                "uom_e":     g["uom"].mode()[0] if len(g) > 0 else uom,
                "sum_qty":   g["Sum_of_QTY"].sum(),
                "total_val": g["Sum_of_TOTAL_VALUE"].sum(),
                "avg_price": _safe_wtd_avg(g["Sum_of_TOTAL_VALUE"], g["Sum_of_QTY"]),
            }))
            .reset_index()
            .sort_values("avg_price")
        )

        exim_rows_html = ""
        for _, row_e in exim_agg.iterrows():
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
              <td>—</td>
              <td>{row_e["grade_e"]}</td>
              <td>{fmt_qty(row_e["sum_qty"])}</td>
              <td>{fmt_inr(row_e["total_val"])}</td>
              <td {price_style}>₹{row_e["avg_price"]:,.0f}</td>
              <td>{vs_html}</td>
            </tr>
            """

        total_qty_e = market_df_f["Sum_of_QTY"].sum()
        total_val_e = market_df_f["Sum_of_TOTAL_VALUE"].sum()
        wtd_avg_e = _safe_wtd_avg(market_df_f["Sum_of_TOTAL_VALUE"], market_df_f["Sum_of_QTY"])
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
          <td>—</td>
          <td>{fmt_qty(total_qty_e)}</td>
          <td>{fmt_inr(total_val_e)}</td>
          <td>₹{wtd_avg_e:,.0f}</td>
          <td>{vs_footer}</td>
        </tr>
        """

        _html(f"""
<div class="pi-card" style="margin-bottom:1.5rem;">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem;">
<div>
<div class="pi-section-title">EXIM — Supplier Price &amp; Volume</div>
<div class="pi-section-sub">supplier/buyer · {period_label} · {uom} · Grade · Sum QTY · Avg PRICE</div>
</div>
<span class="badge badge-cyan">EXIM</span>
</div>
<div style="overflow-x:auto;">
<table class="pi-data-table">
<thead>
<tr>
<th>SUPPLIER</th><th>ORIGIN</th><th>GRADE</th>
<th>SUM OF QTY</th><th>TOTAL VALUE</th><th>AVG PRICE</th><th>VS CIPLA</th>
</tr>
</thead>
<tbody>{exim_rows_html}</tbody>
</table>
</div>
</div>
""")

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
