# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
from pathlib import Path

from src.file_discovery import FileDiscovery
from src.fuzzy_matcher import FuzzyMatcher
from src.pipeline import run_processing_pipeline
from src.utils import Utils
from src.settings import MOLECULE_MAPPING


def _safe_wtd_avg(total_value_series, qty_series) -> float:
    """Return weighted average price; 0.0 when total quantity is zero."""
    total_qty = qty_series.sum()
    if total_qty == 0:
        return 0.0
    return total_value_series.sum() / total_qty


def _fmt_period(yyyymm_str):
    """Convert '202504' -> 'Apr 2025'"""
    try:
        return datetime.strptime(str(yyyymm_str), '%Y%m').strftime('%b %Y')
    except Exception:
        return str(yyyymm_str)


ENTITY_COLOURS = [
    '#0a2342',  # Cipla (navy - always first/special)
    '#00b4d8',  # teal
    '#e63946',  # red
    '#2a9d8f',  # green
    '#e9c46a',  # yellow
    '#f4a261',  # orange
    '#457b9d',  # steel blue
    '#a8dadc',  # light teal
    '#6d6875',  # purple
    '#b5838d',  # rose
    '#264653',  # dark teal
    '#e76f51',  # coral
]


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="PharmaIntel · Price Benchmarking",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS
# ============================================================================
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #f0f4f8;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a2342 0%, #0d2d52 100%);
}
[data-testid="stSidebar"] * { color: #d0dce8 !important; }
[data-testid="stSidebar"] .stButton > button {
    background-color: #1a4a7a;
    color: #ffffff !important;
    border: none;
    border-radius: 6px;
    width: 100%;
    font-weight: 600;
    margin-bottom: 4px;
    transition: background 0.2s;
}
[data-testid="stSidebar"] .stButton > button:hover { background-color: #0d3260; }
[data-testid="stSidebar"] input {
    background-color: #0d2d52 !important;
    color: #d0dce8 !important;
    border: 1px solid #1a4a7a !important;
    border-radius: 6px !important;
}

.pi-header {
    background: linear-gradient(90deg, #0a2342 0%, #0d3260 100%);
    padding: 1.2rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-radius: 10px;
    margin-bottom: 1.4rem;
    box-shadow: 0 4px 14px rgba(0,0,0,0.18);
}
.pi-brand {
    color: #ffffff;
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: 0.3px;
    display: block;
}
.pi-subtitle {
    color: #7fa8cc;
    font-size: 0.78rem;
    display: block;
    margin-top: 2px;
}
.cipla-badge {
    background: #e63946;
    color: #fff;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-right: 8px;
}
.live-badge {
    background: rgba(255,255,255,0.12);
    color: #7fffad;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
}

.search-bar {
    background: #ffffff;
    border-radius: 10px;
    padding: 1.1rem 1.5rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    margin-bottom: 1.4rem;
}

.mol-title-card {
    background: #ffffff;
    border-radius: 10px;
    padding: 1.1rem 1.6rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    margin-bottom: 1.4rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.8rem;
}
.mol-pill {
    background: #0a2342;
    color: #ffffff;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}

.kpi-card {
    background: #ffffff;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    border-left: 4px solid #0a2342;
    height: 100%;
}
.kpi-label {
    font-size: 0.68rem;
    color: #7a8a9a;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 0.35rem;
}
.kpi-value {
    font-size: 1.4rem;
    font-weight: 800;
    color: #0a2342;
    line-height: 1.2;
}
.kpi-delta {
    font-size: 0.72rem;
    margin-top: 0.25rem;
}
.kpi-note {
    font-size: 0.67rem;
    color: #8a9aaa;
    margin-top: 0.3rem;
}
.kpi-navy  { border-left-color: #0a2342; }
.kpi-blue  { border-left-color: #00b4d8; }
.kpi-green { border-left-color: #2a9d8f; }
.kpi-red   { border-left-color: #e63946; }
.kpi-orange{ border-left-color: #f4a261; }

.section-title {
    border-left: 4px solid #0a2342;
    padding-left: 0.75rem;
    font-size: 0.95rem;
    font-weight: 700;
    color: #0a2342;
    margin: 1.4rem 0 0.9rem 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.bench-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}
.bench-table th {
    background: #0a2342;
    color: #ffffff;
    padding: 0.55rem 0.75rem;
    text-align: left;
    font-weight: 600;
    font-size: 0.72rem;
    letter-spacing: 0.5px;
    white-space: nowrap;
}
.bench-table td {
    padding: 0.5rem 0.75rem;
    color: #1a2a3a;
    border-bottom: 1px solid #e8edf5;
    white-space: nowrap;
}
.bench-table tr:nth-child(even) td { background-color: #f8f9fa; }
.bench-table tr.cipla-row td { font-weight: 700; background: #eef3f9 !important; }
.bench-table tr:hover td { background: #e8f0f8 !important; }

.badge-benchmark  { background:#0a2342; color:#fff; padding:2px 10px; border-radius:12px; font-size:0.7rem; font-weight:700; }
.badge-cheaper    { background:#2a9d8f; color:#fff; padding:2px 10px; border-radius:12px; font-size:0.7rem; font-weight:700; }
.badge-slightlyhigher { background:#e9c46a; color:#333; padding:2px 10px; border-radius:12px; font-size:0.7rem; font-weight:700; }
.badge-higher     { background:#f4a261; color:#fff; padding:2px 10px; border-radius:12px; font-size:0.7rem; font-weight:700; }
.badge-premium    { background:#e63946; color:#fff; padding:2px 10px; border-radius:12px; font-size:0.7rem; font-weight:700; }

.simple-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}
.simple-table th {
    background: #eef3f9;
    color: #0a2342;
    padding: 0.5rem 0.75rem;
    text-align: left;
    font-weight: 700;
    font-size: 0.72rem;
    letter-spacing: 0.3px;
    border-bottom: 2px solid #c8d8e8;
}
.simple-table td {
    padding: 0.45rem 0.75rem;
    color: #1a2a3a;
    border-bottom: 1px solid #e8edf5;
}
.simple-table tr:nth-child(even) td { background-color: #f8f9fa; }
.simple-table tr.total-row td { font-weight: 700; background: #eef3f9 !important; border-top: 2px solid #c8d8e8; }

.pi-footer {
    background: #0a2342;
    color: #7a9ab8;
    padding: 1rem 2rem;
    border-radius: 8px;
    margin-top: 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.73rem;
}
.pi-footer div:last-child { color: #aac4de; font-weight: 600; }

.card-wrap {
    background: #ffffff;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    margin-bottom: 1.4rem;
}

.outline-btn {
    display: inline-block;
    border: 1.5px solid #0a2342;
    color: #0a2342;
    padding: 0.3rem 0.9rem;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 8px;
    cursor: pointer;
    text-decoration: none;
}
.outline-btn:hover { background: #eef3f9; }

.analyze-btn > button {
    background: #0d9488 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================
for _k, _v in [
    ('molecule_data', None),
    ('selected_molecule', None),
    ('pipeline_metadata', None),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ============================================================================
# INIT CLASSES
# ============================================================================
file_discovery = FileDiscovery(data_dir="data/raw", molecule_mapping=MOLECULE_MAPPING)
fuzzy_matcher  = FuzzyMatcher(molecule_mapping=MOLECULE_MAPPING, threshold=70)

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div class="pi-header">
  <div class="pi-header-left">
    <span class="pi-brand">PharmaIntel · Price Benchmarking</span>
    <span class="pi-subtitle">Internal EXIM Data Intelligence Platform</span>
  </div>
  <div class="pi-header-right">
    <span class="cipla-badge">CIPLA INTERNAL</span>
    <span class="live-badge">● Live Data</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.markdown("### 🔍 Molecule Search")

available_molecules = file_discovery.get_available_molecules()

search_input = st.sidebar.text_input(
    "Search molecule:",
    placeholder="e.g., azithromycoon, cipro...",
    help="Supports fuzzy matching for typos and aliases",
)

matches = []
if search_input.strip():
    matches = fuzzy_matcher.match_molecule_input(search_input)

if matches:
    st.sidebar.markdown("**🎯 Suggestions**")
    for mol_name, confidence in matches:
        if mol_name in available_molecules:
            st.sidebar.caption(f"{mol_name.upper()}  ·  {confidence}% match")
            if st.sidebar.button(f"Select  {mol_name.upper()}", key=f"sb_btn_{mol_name}"):
                st.session_state.selected_molecule = mol_name
                st.rerun()
elif search_input.strip():
    st.sidebar.warning("No matches found — try a different spelling.")

st.sidebar.markdown("---")
st.sidebar.markdown("**📋 Available Molecules**")
for mol_name, mol_info in available_molecules.items():
    if st.sidebar.button(f"💊  {mol_name.upper()}", key=f"sb_mol_{mol_name}"):
        st.session_state.selected_molecule = mol_name
        st.rerun()

if st.session_state.selected_molecule:
    st.sidebar.markdown("---")
    st.sidebar.success(f"✅ Loaded: **{st.session_state.selected_molecule.upper()}**")

# ============================================================================
# HELPERS
# ============================================================================

def _position_badge(vs_pct):
    """Return HTML badge string for a vs-Cipla percentage."""
    if vs_pct < -5:
        return '<span class="badge-cheaper">Cheaper</span>'
    elif vs_pct <= 5:
        return '<span class="badge-slightlyhigher">Slightly Higher</span>'
    elif vs_pct <= 20:
        return '<span class="badge-higher">Higher</span>'
    else:
        return '<span class="badge-premium">Premium</span>'


def _entity_colour_map(entity_list, cipla_entity_names=None):
    """Return dict mapping entity name -> colour, Cipla entities always navy."""
    if cipla_entity_names is None:
        cipla_entity_names = set()
    cmap = {}
    palette_idx = 1  # index 0 is navy for Cipla
    for ent in entity_list:
        if ent not in cmap:
            is_cipla = (
                ent in cipla_entity_names
                or ent.lower().startswith('cipla')
                or ent.lower() == 'cipla'
            )
            if is_cipla:
                cmap[ent] = ENTITY_COLOURS[0]
            else:
                cmap[ent] = ENTITY_COLOURS[palette_idx % len(ENTITY_COLOURS)]
                palette_idx += 1
    return cmap


# ============================================================================
# MAIN CONTENT
# ============================================================================

if st.session_state.selected_molecule:
    selected_mol = st.session_state.selected_molecule

    with st.spinner(f"Loading data for {selected_mol.upper()}…"):
        result = run_processing_pipeline(selected_mol, file_discovery)

    if result['status'] == 'failed':
        st.error(f"❌ Pipeline failed: {', '.join(result['errors'])}")
        st.stop()

    metadata        = result['metadata']
    cipla_baseline  = metadata['cipla_baseline']
    filter_stats    = metadata['filter_stats']
    consolidated_df = result['data']['consolidated']

    # Build set of entity names that belong to Cipla source
    cipla_entity_names = set(
        consolidated_df.loc[consolidated_df['source'] == 'Cipla', 'entity_name'].unique()
    )

    # Mol info
    mol_cfg   = MOLECULE_MAPPING['molecules'].get(selected_mol, {})
    cas_code  = mol_cfg.get('cipla_api_filter', selected_mol.upper())
    mol_desc  = mol_cfg.get('description', '')

    # ── Filter Bar ────────────────────────────────────────────────────────────
    st.markdown('<div class="search-bar">', unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.7rem;font-weight:700;color:#7a8a9a;"
        "letter-spacing:1px;margin-bottom:0.6rem;'>MATERIAL INTELLIGENCE SEARCH</p>",
        unsafe_allow_html=True,
    )
    fb1, fb2, fb3, fb4 = st.columns([2, 2, 2, 1])

    with fb1:
        st.text_input("Molecule", value=selected_mol.upper(), disabled=True, key="fi_mol")

    period_opts = ["Last 3 Months", "Last 6 Months", "Last 12 Months", "All Time"]
    with fb2:
        period_sel = st.selectbox("Period", period_opts, index=1, key="fi_period")

    source_opts = ["All", "Supplier", "Buyer"]
    with fb3:
        source_sel = st.selectbox("Origin / Source", source_opts, key="fi_source")

    with fb4:
        st.markdown('<div class="analyze-btn">', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        analyze = st.button("Analyze Now ▶", key="fi_analyze")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Apply period filter
    all_periods = sorted(consolidated_df['yyyymm'].unique())
    period_map  = {"Last 3 Months": 3, "Last 6 Months": 6, "Last 12 Months": 12}
    if period_sel in period_map:
        n = period_map[period_sel]
        keep_periods = all_periods[-n:] if len(all_periods) >= n else all_periods
    else:
        keep_periods = all_periods

    filtered_df = consolidated_df[consolidated_df['yyyymm'].isin(keep_periods)].copy()

    # Apply source filter
    if source_sel != "All":
        filtered_df = filtered_df[
            (filtered_df['source'] == source_sel) | (filtered_df['source'] == 'Cipla')
        ]

    max_period_str = _fmt_period(keep_periods[-1]) if keep_periods else "N/A"

    # ── Molecule Title Card ───────────────────────────────────────────────────
    st.markdown(f"""
    <div class="mol-title-card">
      <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">
        <span class="mol-pill">💊 {selected_mol.upper()}</span>
        <span style="font-size:0.82rem;color:#4a5a6a;">API · {cas_code}</span>
        <span style="font-size:0.82rem;color:#4a5a6a;">HS Code: —</span>
        <span style="font-size:0.82rem;color:#4a5a6a;">Unit: INR/KG</span>
        <span style="font-size:0.82rem;color:#4a5a6a;">Last Updated: {max_period_str}</span>
      </div>
      <div>
        <span class="outline-btn">📄 Export PDF</span>
        <span class="outline-btn">📊 Export Excel</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Computations ─────────────────────────────────────────────────────
    cipla_rows   = filtered_df[filtered_df['source'] == 'Cipla']
    market_rows  = filtered_df[filtered_df['source'] != 'Cipla']

    cipla_price  = _safe_wtd_avg(cipla_rows['Sum_of_TOTAL_VALUE'], cipla_rows['Sum_of_QTY'])
    market_price = _safe_wtd_avg(market_rows['Sum_of_TOTAL_VALUE'], market_rows['Sum_of_QTY'])
    cipla_n      = len(cipla_rows)
    market_n_ent = market_rows['entity_name'].nunique()

    # Per-entity weighted avg for min/max
    if len(market_rows) > 0:
        ent_prices = (
            market_rows.groupby('entity_name')
            .apply(lambda g: _safe_wtd_avg(g['Sum_of_TOTAL_VALUE'], g['Sum_of_QTY']))
            .reset_index(name='wtd_price')
        )
        ent_prices = ent_prices[ent_prices['wtd_price'] > 0]
        if len(ent_prices) > 0:
            min_row   = ent_prices.loc[ent_prices['wtd_price'].idxmin()]
            max_row   = ent_prices.loc[ent_prices['wtd_price'].idxmax()]
            low_price = min_row['wtd_price']
            low_ent   = min_row['entity_name']
            high_price= max_row['wtd_price']
            high_ent  = max_row['entity_name']
        else:
            low_price = high_price = 0.0
            low_ent   = high_ent  = "—"
    else:
        low_price = high_price = 0.0
        low_ent   = high_ent  = "—"

    cost_adv     = cipla_price - market_price if market_price > 0 else 0.0
    cost_adv_col = "#2a9d8f" if cost_adv <= 0 else "#e63946"

    # Previous period delta
    prev_cipla_price  = 0.0
    prev_market_price = 0.0
    if len(keep_periods) >= 2:
        prev_period_list = keep_periods[:-1]
        prev_df = consolidated_df[consolidated_df['yyyymm'].isin(prev_period_list)]
        prev_c  = prev_df[prev_df['source'] == 'Cipla']
        prev_m  = prev_df[prev_df['source'] != 'Cipla']
        prev_cipla_price  = _safe_wtd_avg(prev_c['Sum_of_TOTAL_VALUE'], prev_c['Sum_of_QTY'])
        prev_market_price = _safe_wtd_avg(prev_m['Sum_of_TOTAL_VALUE'], prev_m['Sum_of_QTY'])

    def _delta_html(curr, prev):
        if prev == 0:
            return ""
        diff = curr - prev
        sign = "▲" if diff > 0 else "▼"
        col  = "#e63946" if diff > 0 else "#2a9d8f"
        return (
            f'<div class="kpi-delta" style="color:{col};">'
            f'{sign} ₹{abs(diff):,.0f} vs prev period</div>'
        )

    # ── 5 KPI Cards ───────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.markdown(f"""
        <div class="kpi-card kpi-navy">
          <div class="kpi-label">Cipla WTD Avg Price</div>
          <div class="kpi-value">₹{cipla_price:,.0f} <span style="font-size:0.75rem;font-weight:400;">/KG</span></div>
          {_delta_html(cipla_price, prev_cipla_price)}
          <div class="kpi-note">Weighted across {cipla_n} records</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi-card kpi-blue">
          <div class="kpi-label">Market Avg (EXIM)</div>
          <div class="kpi-value">₹{market_price:,.0f} <span style="font-size:0.75rem;font-weight:400;">/KG</span></div>
          {_delta_html(market_price, prev_market_price)}
          <div class="kpi-note">Across {market_n_ent} entities</div>
        </div>
        """, unsafe_allow_html=True)

    cost_sign = "▼" if cost_adv < 0 else "▲"
    with k3:
        st.markdown(f"""
        <div class="kpi-card kpi-green">
          <div class="kpi-label">Cost Advantage</div>
          <div class="kpi-value" style="color:{cost_adv_col};">{cost_sign} ₹{abs(cost_adv):,.0f}</div>
          <div class="kpi-note">vs Market EXIM Average</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="kpi-card kpi-red">
          <div class="kpi-label">Lowest Competitor</div>
          <div class="kpi-value">₹{low_price:,.0f} <span style="font-size:0.75rem;font-weight:400;">/KG</span></div>
          <div class="kpi-note">{low_ent[:28] if low_ent else '—'}</div>
        </div>
        """, unsafe_allow_html=True)

    with k5:
        st.markdown(f"""
        <div class="kpi-card kpi-orange">
          <div class="kpi-label">Highest Competitor</div>
          <div class="kpi-value">₹{high_price:,.0f} <span style="font-size:0.75rem;font-weight:400;">/KG</span></div>
          <div class="kpi-note">{high_ent[:28] if high_ent else '—'}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ========================================================================
    # SECTION 2 – Competitor Benchmark Line Chart
    # ========================================================================
    st.markdown('<div class="card-wrap">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">| Competitor Price Benchmark — Last 12 Months</div>',
        unsafe_allow_html=True,
    )

    # Build per-entity per-period WTD avg
    chart_df = (
        filtered_df.groupby(['yyyymm', 'entity_name'])
        .apply(lambda g: _safe_wtd_avg(g['Sum_of_TOTAL_VALUE'], g['Sum_of_QTY']))
        .reset_index(name='wtd_price')
    )

    # Market avg per period (non-Cipla)
    mkt_per_period = (
        filtered_df[filtered_df['source'] != 'Cipla']
        .groupby('yyyymm')
        .apply(lambda g: _safe_wtd_avg(g['Sum_of_TOTAL_VALUE'], g['Sum_of_QTY']))
        .reset_index(name='wtd_price')
    )
    mkt_per_period['entity_name'] = 'Market Avg (EXIM)'

    chart_all = pd.concat([chart_df, mkt_per_period], ignore_index=True)

    all_entities = sorted(chart_df['entity_name'].unique().tolist())
    cmap = _entity_colour_map(all_entities, cipla_entity_names)
    cmap['Market Avg (EXIM)'] = '#888888'

    fig_bench = go.Figure()

    for ent in all_entities:
        sub = chart_df[chart_df['entity_name'] == ent].sort_values('yyyymm')
        sub['period_label'] = sub['yyyymm'].apply(_fmt_period)
        is_cipla = ent in cipla_entity_names
        colour   = cmap.get(ent, ENTITY_COLOURS[1])
        fig_bench.add_trace(go.Scatter(
            x=sub['period_label'],
            y=sub['wtd_price'],
            mode='lines+markers',
            name=ent,
            line=dict(
                color=colour,
                width=3 if is_cipla else 1.8,
            ),
            marker=dict(size=7 if is_cipla else 5, color=colour),
        ))

    if len(mkt_per_period) > 0:
        mkt_sub = mkt_per_period.sort_values('yyyymm').copy()
        mkt_sub['period_label'] = mkt_sub['yyyymm'].apply(_fmt_period)
        fig_bench.add_trace(go.Scatter(
            x=mkt_sub['period_label'],
            y=mkt_sub['wtd_price'],
            mode='lines',
            name='Market Avg (EXIM)',
            line=dict(color='#888888', width=2, dash='dash'),
        ))

    fig_bench.update_layout(
        height=380,
        paper_bgcolor='white',
        plot_bgcolor='#f8fbff',
        xaxis_title="Period",
        yaxis_title="WTD Avg Price (₹/KG)",
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(size=11),
    )
    st.plotly_chart(fig_bench, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ========================================================================
    # SECTION 3 – Bubble + Donut Row
    # ========================================================================
    bc_left, bc_right = st.columns(2)

    # Per-entity aggregates for bubble
    bubble_agg = (
        filtered_df.groupby('entity_name')
        .apply(lambda g: pd.Series({
            'avg_price':  _safe_wtd_avg(g['Sum_of_TOTAL_VALUE'], g['Sum_of_QTY']),
            'total_qty':  g['Sum_of_QTY'].sum(),
            'shipments':  len(g),
            'source':     g['source'].iloc[0],
        }))
        .reset_index()
    )
    bubble_agg = bubble_agg[bubble_agg['avg_price'] > 0]
    bubble_agg['volume_mt'] = bubble_agg['total_qty'] / 1000.0

    all_bubble_ents = sorted(bubble_agg['entity_name'].tolist())
    b_cmap = _entity_colour_map(all_bubble_ents, cipla_entity_names)

    with bc_left:
        st.markdown('<div class="card-wrap">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">| Bubble Analysis · Price vs Volume vs Frequency</div>',
            unsafe_allow_html=True,
        )
        fig_bub = go.Figure()
        for ent in all_bubble_ents:
            row = bubble_agg[bubble_agg['entity_name'] == ent]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            fig_bub.add_trace(go.Scatter(
                x=[r['avg_price']],
                y=[r['volume_mt']],
                mode='markers',
                name=ent,
                marker=dict(
                    size=max(8, min(60, r['shipments'] * 5)),
                    color=b_cmap.get(ent, ENTITY_COLOURS[1]),
                    opacity=0.8,
                    line=dict(width=1, color='white'),
                ),
                text=f"{ent}<br>Shipments: {int(r['shipments'])}",
                hovertemplate="%{text}<br>Avg Price: ₹%{x:,.0f}<br>Volume: %{y:.1f} MT<extra></extra>",
            ))
        fig_bub.update_layout(
            height=360,
            xaxis_title="Avg Price (₹/KG)",
            yaxis_title="Volume (MT)",
            paper_bgcolor='white',
            plot_bgcolor='#f8fbff',
            legend=dict(orientation='v', x=1.02, y=1),
            margin=dict(l=10, r=10, t=10, b=10),
            font=dict(size=11),
        )
        st.plotly_chart(fig_bub, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with bc_right:
        st.markdown('<div class="card-wrap">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">| Volume Share by Competitor</div>',
            unsafe_allow_html=True,
        )
        donut_labels = bubble_agg['entity_name'].tolist()
        donut_values = bubble_agg['volume_mt'].tolist()
        donut_colours = [b_cmap.get(e, ENTITY_COLOURS[1]) for e in donut_labels]

        fig_donut = go.Figure(go.Pie(
            labels=donut_labels,
            values=donut_values,
            hole=0.6,
            marker=dict(colors=donut_colours),
            textinfo='percent',
            hovertemplate="%{label}<br>%{value:.1f} MT (%{percent})<extra></extra>",
        ))
        fig_donut.update_layout(
            height=360,
            paper_bgcolor='white',
            legend=dict(orientation='v', x=1.02, y=1),
            margin=dict(l=10, r=10, t=10, b=10),
            font=dict(size=11),
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ========================================================================
    # BENCHMARK TABLE
    # ========================================================================
    st.markdown('<div class="card-wrap">', unsafe_allow_html=True)
    period_note = f"{_fmt_period(keep_periods[0])} – {_fmt_period(keep_periods[-1])}" if keep_periods else "All Time"
    st.markdown(
        f'<div class="section-title">| Competitor Benchmark Table'
        f'<span style="font-size:0.72rem;font-weight:400;color:#7a8a9a;">'
        f'  {period_note} · Source: {source_sel}</span></div>',
        unsafe_allow_html=True,
    )

    bench_agg = (
        filtered_df.groupby('entity_name')
        .apply(lambda g: pd.Series({
            'avg_price':  g['Avg_PRICE'].mean(),
            'wtd_price':  _safe_wtd_avg(g['Sum_of_TOTAL_VALUE'], g['Sum_of_QTY']),
            'total_qty':  g['Sum_of_QTY'].sum(),
            'shipments':  len(g),
            'source':     g['source'].iloc[0],
        }))
        .reset_index()
    )

    # Sort: Cipla first, rest by wtd_price ascending
    cipla_mask   = bench_agg['source'] == 'Cipla'
    bench_cipla  = bench_agg[cipla_mask].copy()
    bench_others = bench_agg[~cipla_mask].sort_values('wtd_price').reset_index(drop=True)
    bench_sorted = pd.concat([bench_cipla, bench_others], ignore_index=True)

    cipla_ref_price = bench_cipla['wtd_price'].mean() if len(bench_cipla) > 0 else cipla_baseline['avg_price']

    rows_html   = ""
    other_count = 0
    for _, row in bench_sorted.iterrows():
        is_cipla_row = row['source'] == 'Cipla'
        row_cls      = "cipla-row" if is_cipla_row else ""
        if is_cipla_row:
            num_str = ""
        else:
            other_count += 1
            num_str = str(other_count)

        vs_pct_val = ((row['wtd_price'] - cipla_ref_price) / cipla_ref_price * 100) if cipla_ref_price > 0 else 0.0

        if is_cipla_row:
            badge_html = '<span class="badge-benchmark">Benchmark</span>'
            vs_str     = "—"
        else:
            badge_html = _position_badge(vs_pct_val)
            sign_sym   = "▼" if vs_pct_val < 0 else "▲"
            sign_col   = "#2a9d8f" if vs_pct_val < 0 else "#e63946"
            vs_str     = (
                f'<span style="color:{sign_col};font-weight:700;">'
                f'{sign_sym} {abs(vs_pct_val):.1f}%</span>'
            )

        rows_html += f"""
        <tr class="{row_cls}">
          <td>{num_str}</td>
          <td><strong>{row['entity_name']}</strong></td>
          <td>{row['source']}</td>
          <td>₹{row['avg_price']:,.0f}</td>
          <td>₹{row['wtd_price']:,.0f}</td>
          <td>{row['total_qty']/1000:.1f}</td>
          <td>{int(row['shipments'])}</td>
          <td>{vs_str}</td>
          <td>{badge_html}</td>
        </tr>
        """

    table_html = f"""
    <table class="bench-table">
      <thead>
        <tr>
          <th>#</th>
          <th>COMPANY</th>
          <th>ORIGIN</th>
          <th>AVG PRICE (₹/KG)</th>
          <th>WTD AVG PRICE</th>
          <th>TOTAL VOLUME (MT)</th>
          <th>NO. SHIPMENTS</th>
          <th>VS CIPLA (%)</th>
          <th>POSITION</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ========================================================================
    # SECTION 4 – Volume & Price Tables (side by side)
    # ========================================================================
    vt_left, vt_right = st.columns(2)

    with vt_left:
        st.markdown('<div class="card-wrap">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">| Volume Data (MT)</div>',
            unsafe_allow_html=True,
        )

        vol_rows = []
        for per in keep_periods:
            per_df   = filtered_df[filtered_df['yyyymm'] == per]
            cip_vol  = per_df[per_df['source'] == 'Cipla']['Sum_of_QTY'].sum() / 1000
            mkt_vol  = per_df[per_df['source'] != 'Cipla']['Sum_of_QTY'].sum() / 1000
            total    = cip_vol + mkt_vol
            share    = (cip_vol / total * 100) if total > 0 else 0.0
            vol_rows.append({
                'period':   per,
                'cipla_mt': cip_vol,
                'mkt_mt':   mkt_vol,
                'share':    share,
            })

        vol_body = ""
        for r in vol_rows:
            vol_body += (
                f"<tr><td>{_fmt_period(r['period'])}</td>"
                f"<td>{r['cipla_mt']:.1f}</td>"
                f"<td>{r['mkt_mt']:.1f}</td>"
                f"<td>{r['share']:.1f}%</td></tr>"
            )
        # Totals
        t_cip = sum(r['cipla_mt'] for r in vol_rows)
        t_mkt = sum(r['mkt_mt']   for r in vol_rows)
        t_tot = t_cip + t_mkt
        t_shr = (t_cip / t_tot * 100) if t_tot > 0 else 0.0
        vol_html = f"""
        <table class="simple-table">
          <thead>
            <tr><th>MONTH</th><th>CIPLA (MT)</th><th>MARKET (MT)</th><th>CIPLA SHARE (%)</th></tr>
          </thead>
          <tbody>
            {vol_body}
            <tr class="total-row">
              <td>TOTAL</td>
              <td>{t_cip:.1f}</td>
              <td>{t_mkt:.1f}</td>
              <td>{t_shr:.1f}%</td>
            </tr>
          </tbody>
        </table>
        """
        st.markdown(vol_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with vt_right:
        st.markdown('<div class="card-wrap">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">| Price Trend (₹/KG)</div>',
            unsafe_allow_html=True,
        )

        price_rows = []
        for per in keep_periods:
            per_df    = filtered_df[filtered_df['yyyymm'] == per]
            cip_pr    = _safe_wtd_avg(
                per_df[per_df['source'] == 'Cipla']['Sum_of_TOTAL_VALUE'],
                per_df[per_df['source'] == 'Cipla']['Sum_of_QTY'],
            )
            mkt_pr    = _safe_wtd_avg(
                per_df[per_df['source'] != 'Cipla']['Sum_of_TOTAL_VALUE'],
                per_df[per_df['source'] != 'Cipla']['Sum_of_QTY'],
            )
            spread    = cip_pr - mkt_pr
            price_rows.append({'period': per, 'cipla': cip_pr, 'mkt': mkt_pr, 'spread': spread})

        price_body = ""
        for r in price_rows:
            sprd_col  = "#2a9d8f" if r['spread'] < 0 else "#e63946"
            sprd_sign = "▼" if r['spread'] < 0 else "▲"
            price_body += (
                f"<tr><td>{_fmt_period(r['period'])}</td>"
                f"<td>₹{r['cipla']:,.0f}</td>"
                f"<td>₹{r['mkt']:,.0f}</td>"
                f"<td style='color:{sprd_col};font-weight:700;'>"
                f"{sprd_sign} ₹{abs(r['spread']):,.0f}</td></tr>"
            )
        # WTD Avg footer
        all_cip  = filtered_df[filtered_df['source'] == 'Cipla']
        all_mkt  = filtered_df[filtered_df['source'] != 'Cipla']
        wtd_cip  = _safe_wtd_avg(all_cip['Sum_of_TOTAL_VALUE'], all_cip['Sum_of_QTY'])
        wtd_mkt  = _safe_wtd_avg(all_mkt['Sum_of_TOTAL_VALUE'], all_mkt['Sum_of_QTY'])
        wtd_sprd = wtd_cip - wtd_mkt
        sprd_col = "#2a9d8f" if wtd_sprd < 0 else "#e63946"
        sprd_sym = "▼" if wtd_sprd < 0 else "▲"

        price_html = f"""
        <table class="simple-table">
          <thead>
            <tr><th>MONTH</th><th>CIPLA AVG</th><th>MARKET AVG</th><th>SPREAD</th></tr>
          </thead>
          <tbody>
            {price_body}
            <tr class="total-row">
              <td>WTD AVG</td>
              <td>₹{wtd_cip:,.0f}</td>
              <td>₹{wtd_mkt:,.0f}</td>
              <td style='color:{sprd_col};font-weight:700;'>{sprd_sym} ₹{abs(wtd_sprd):,.0f}</td>
            </tr>
          </tbody>
        </table>
        """
        st.markdown(price_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ========================================================================
    # SECTION 5 – Exact Grade Match
    # ========================================================================
    st.markdown('<div class="card-wrap">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">5. Exact Grade Match</div>',
        unsafe_allow_html=True,
    )

    cipla_grade_rows = filtered_df[filtered_df['source'] == 'Cipla']
    if len(cipla_grade_rows) > 0:
        cipla_primary_grade = (
            cipla_grade_rows['GRADE_SPEC']
            .value_counts()
            .idxmax()
        )
        grade_filtered_df = filtered_df[filtered_df['GRADE_SPEC'] == cipla_primary_grade]

        st.info(
            f"Showing only transactions matching Cipla's primary grade: **{cipla_primary_grade}**"
        )

        grade_bench = (
            grade_filtered_df.groupby('entity_name')
            .apply(lambda g: pd.Series({
                'avg_price': g['Avg_PRICE'].mean(),
                'wtd_price': _safe_wtd_avg(g['Sum_of_TOTAL_VALUE'], g['Sum_of_QTY']),
                'total_qty': g['Sum_of_QTY'].sum(),
                'shipments': len(g),
                'source':    g['source'].iloc[0],
            }))
            .reset_index()
        )

        g_cipla_mask  = grade_bench['source'] == 'Cipla'
        g_bench_cipla = grade_bench[g_cipla_mask].copy()
        g_bench_other = grade_bench[~g_cipla_mask].sort_values('wtd_price').reset_index(drop=True)
        g_bench_sort  = pd.concat([g_bench_cipla, g_bench_other], ignore_index=True)
        g_cipla_price = g_bench_cipla['wtd_price'].mean() if len(g_bench_cipla) > 0 else cipla_ref_price

        g_rows_html  = ""
        g_other_cnt  = 0
        for _, row in g_bench_sort.iterrows():
            is_c    = row['source'] == 'Cipla'
            row_cls = "cipla-row" if is_c else ""
            if is_c:
                num_str = ""
            else:
                g_other_cnt += 1
                num_str = str(g_other_cnt)
            vs_pct_v  = ((row['wtd_price'] - g_cipla_price) / g_cipla_price * 100) if g_cipla_price > 0 else 0.0
            if is_c:
                badge_h = '<span class="badge-benchmark">Benchmark</span>'
                vs_s    = "—"
            else:
                badge_h = _position_badge(vs_pct_v)
                s_sym   = "▼" if vs_pct_v < 0 else "▲"
                s_col   = "#2a9d8f" if vs_pct_v < 0 else "#e63946"
                vs_s    = (
                    f'<span style="color:{s_col};font-weight:700;">'
                    f'{s_sym} {abs(vs_pct_v):.1f}%</span>'
                )
            g_rows_html += f"""
            <tr class="{row_cls}">
              <td>{num_str}</td>
              <td><strong>{row['entity_name']}</strong></td>
              <td>{row['source']}</td>
              <td>₹{row['avg_price']:,.0f}</td>
              <td>₹{row['wtd_price']:,.0f}</td>
              <td>{row['total_qty']/1000:.1f}</td>
              <td>{int(row['shipments'])}</td>
              <td>{vs_s}</td>
              <td>{badge_h}</td>
            </tr>
            """

        g_table_html = f"""
        <table class="bench-table">
          <thead>
            <tr>
              <th>#</th><th>COMPANY</th><th>ORIGIN</th>
              <th>AVG PRICE (₹/KG)</th><th>WTD AVG PRICE</th>
              <th>TOTAL VOLUME (MT)</th><th>NO. SHIPMENTS</th>
              <th>VS CIPLA (%)</th><th>POSITION</th>
            </tr>
          </thead>
          <tbody>{g_rows_html}</tbody>
        </table>
        """
        st.markdown(g_table_html, unsafe_allow_html=True)
    else:
        st.info("No Cipla rows in the filtered data — cannot determine primary grade.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ========================================================================
    # FOOTER
    # ========================================================================
    st.markdown("""
    <div class="pi-footer">
      <div>PharmaIntel | Price Benchmarking Intelligence Platform |
           Data Sources: Internal ERP EXIM Trade Data |
           Confidential — For Internal Use Only</div>
      <div>© 2026 Cipla Ltd</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# LANDING PAGE (no molecule selected)
# ============================================================================
else:
    st.markdown("""
    <div class="card-wrap">
      <h2 style="color:#0a2342;margin-top:0;">👋 Welcome to PharmaIntel · Price Benchmarking</h2>
      <p style="color:#4a5a6a;font-size:0.95rem;">
        Select a molecule from the sidebar to explore Cipla's procurement intelligence across
        <strong>Suppliers</strong>, <strong>Buyers</strong>, and <strong>Cipla's own GRN data</strong>.
      </p>
      <hr style="border-top:1px solid #dde3ee;">
      <p style="color:#0a2342;font-weight:700;margin-bottom:0.5rem;">This dashboard delivers:</p>
      <ul style="color:#4a5a6a;font-size:0.9rem;line-height:1.9;">
        <li>KPI summary — Cipla WTD avg price vs market average, cost advantage, min/max competitors</li>
        <li>Competitor benchmark line chart — entity-level price trends over time</li>
        <li>Bubble analysis — price vs volume vs shipment frequency</li>
        <li>Volume share donut — who controls the market volume</li>
        <li>Styled benchmark table — rank-ordered with vs-Cipla % and position badges</li>
        <li>Volume & price monthly tables — period-by-period breakdown</li>
        <li>Exact grade match — filtered to Cipla's primary GRADE_SPEC</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

    available_molecules = file_discovery.get_available_molecules()
    if available_molecules:
        st.markdown(
            '<div class="section-title" style="margin-top:1.5rem;">| Available Molecules</div>',
            unsafe_allow_html=True,
        )
        mol_cols = st.columns(min(len(available_molecules), 4))
        for idx, (mol_name, mol_info) in enumerate(available_molecules.items()):
            with mol_cols[idx % 4]:
                st.markdown(f"""
                <div class="kpi-card kpi-navy" style="text-align:center;padding:1.2rem;margin-bottom:8px;">
                  <div class="kpi-label">Molecule</div>
                  <div class="kpi-value" style="font-size:1rem;">{mol_name.upper()}</div>
                  <div style="font-size:0.7rem;color:#7a8a9a;margin-top:0.3rem;">{mol_info.get('description','')}</div>
                  <div style="font-size:0.7rem;color:#9aabb8;margin-top:0.2rem;">
                    {mol_info.get('file_count',0)} file(s) &nbsp;|&nbsp;
                    {'✓ Cipla data' if mol_info.get('cipla_available') else '✗ No Cipla'}
                  </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Load {mol_name.upper()} →", key=f"land_{mol_name}"):
                    st.session_state.selected_molecule = mol_name
                    st.rerun()

    st.markdown("""
    <div class="pi-footer" style="margin-top:2rem;">
      <div>PharmaIntel | Price Benchmarking Intelligence Platform |
           Data Sources: Internal ERP EXIM Trade Data |
           Confidential — For Internal Use Only</div>
      <div>© 2026 Cipla Ltd</div>
    </div>
    """, unsafe_allow_html=True)
