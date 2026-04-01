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


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Pharma Price Benchmarking",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* ── Global ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #f4f6fb;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a3a5c 0%, #1f5080 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e8edf5 !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background-color: #2e86c1;
        color: #ffffff !important;
        border: none;
        border-radius: 6px;
        width: 100%;
        font-weight: 600;
        transition: background 0.2s;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #1a6fa8;
    }
    [data-testid="stSidebar"] input {
        background-color: #254f73 !important;
        color: #e8edf5 !important;
        border: 1px solid #3a7ab5 !important;
        border-radius: 6px !important;
    }

    /* ── Top header banner ── */
    .app-header {
        background: linear-gradient(90deg, #1a3a5c 0%, #2e86c1 100%);
        padding: 1.4rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .app-header h1 {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.5px;
    }
    .app-header .subtitle {
        color: #b3d1e8;
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }
    .header-badge {
        background: rgba(255,255,255,0.15);
        color: #fff;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* ── KPI / metric cards ── */
    .kpi-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
    }
    .kpi-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 1.1rem 1.4rem;
        flex: 1;
        min-width: 140px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border-left: 4px solid #2e86c1;
    }
    .kpi-card .kpi-label {
        font-size: 0.72rem;
        color: #6b7a8d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }
    .kpi-card .kpi-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #1a3a5c;
    }
    .kpi-card .kpi-delta {
        font-size: 0.75rem;
        color: #e74c3c;
        margin-top: 0.2rem;
    }
    .kpi-card.accent { border-left-color: #27ae60; }
    .kpi-card.warn   { border-left-color: #e67e22; }
    .kpi-card.info   { border-left-color: #8e44ad; }

    /* ── Section header strip ── */
    .section-header {
        background: linear-gradient(90deg, #1a3a5c, #2e86c1);
        color: #ffffff !important;
        padding: 0.65rem 1.2rem;
        border-radius: 8px;
        font-size: 1.05rem;
        font-weight: 700;
        margin: 1.6rem 0 1rem 0;
        letter-spacing: 0.3px;
    }

    /* ── Filter bar ── */
    .filter-bar {
        background: #ffffff;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 1.5rem;
    }

    /* ── Chart card ── */
    .chart-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 1.5rem;
    }

    /* ── Table card ── */
    .table-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 1.5rem;
    }
    .table-card h4 {
        color: #1a3a5c;
        margin-bottom: 0.8rem;
        font-size: 0.95rem;
        font-weight: 700;
        border-bottom: 2px solid #e8edf5;
        padding-bottom: 0.4rem;
    }

    /* ── Cipla highlight card ── */
    .cipla-highlight {
        background: linear-gradient(135deg, #1a3a5c 0%, #2e86c1 100%);
        color: #ffffff;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        text-align: center;
        box-shadow: 0 4px 14px rgba(30,80,140,0.3);
    }
    .cipla-highlight .main-price {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -1px;
        line-height: 1.1;
    }
    .cipla-highlight .price-label {
        font-size: 0.85rem;
        color: #b3d1e8;
        margin-top: 0.3rem;
    }

    /* ── Landing page cards ── */
    .welcome-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }
    .step-item {
        display: flex;
        align-items: flex-start;
        gap: 0.9rem;
        margin-bottom: 0.9rem;
    }
    .step-num {
        background: #2e86c1;
        color: #fff;
        width: 26px;
        height: 26px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.8rem;
        flex-shrink: 0;
    }
    .step-text {
        color: #333;
        font-size: 0.92rem;
        padding-top: 3px;
    }

    /* ── Divider ── */
    hr { border: none; border-top: 1px solid #dde3ee; margin: 1rem 0; }

    /* ── Streamlit default overrides ── */
    [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
    div[data-testid="stHorizontalBlock"] > div { gap: 0.6rem; }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'molecule_data' not in st.session_state:
    st.session_state.molecule_data = None
if 'selected_molecule' not in st.session_state:
    st.session_state.selected_molecule = None
if 'pipeline_metadata' not in st.session_state:
    st.session_state.pipeline_metadata = None

# ============================================================================
# INITIALIZE CLASSES
# ============================================================================

file_discovery = FileDiscovery(
    data_dir="data/raw",
    molecule_mapping=MOLECULE_MAPPING
)
fuzzy_matcher = FuzzyMatcher(molecule_mapping=MOLECULE_MAPPING, threshold=70)

# ============================================================================
# HEADER BANNER
# ============================================================================
current_date = datetime.now().strftime("%d %b %Y")
st.markdown(f"""
<div class="app-header">
  <div>
    <h1>💊 Pharmaceutical Price Benchmarking</h1>
    <div class="subtitle">Cipla Internal Procurement Intelligence Platform</div>
  </div>
  <div class="header-badge">📅 {current_date}</div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - MOLECULE SELECTION
# ============================================================================
st.sidebar.markdown("## 🔍 Molecule Selection")

available_molecules = file_discovery.get_available_molecules()

# Search box with fuzzy matching
search_input = st.sidebar.text_input(
    "Search molecule:",
    placeholder="e.g., azithromycoon, cipro...",
    help="Supports fuzzy matching for typos and molecule aliases"
)

matches = []
if search_input.strip():
    matches = fuzzy_matcher.match_molecule_input(search_input)

# Display search results
if matches:
    st.sidebar.markdown("**🎯 Suggestions**")
    for mol_name, confidence in matches:
        if mol_name in available_molecules:
            c1, c2 = st.sidebar.columns([3, 1])
            with c1:
                st.write(f"**{mol_name.upper()}**")
            with c2:
                st.caption(f"{confidence}%")
            if st.sidebar.button(f"Select {mol_name.upper()}", key=f"btn_{mol_name}"):
                st.session_state.selected_molecule = mol_name
                st.rerun()
elif search_input.strip():
    st.sidebar.warning("No matches found — try a different spelling.")

# Display available molecules
st.sidebar.markdown("---")
st.sidebar.markdown("**📋 Available Molecules**")

for mol_name, mol_info in available_molecules.items():
    with st.sidebar.expander(f"✓ {mol_name.upper()}", expanded=False):
        st.write(f"**Description:** {mol_info['description']}")
        st.write(f"**Files:** {mol_info['file_count']}")
        st.write(f"**Cipla Data:** {'✓ Yes' if mol_info['cipla_available'] else '✗ No'}")
        if st.button("Load →", key=f"load_{mol_name}"):
            st.session_state.selected_molecule = mol_name
            st.rerun()

# ============================================================================
# MAIN CONTENT
# ============================================================================

if st.session_state.selected_molecule:
    selected_mol = st.session_state.selected_molecule

    st.sidebar.markdown("---")
    st.sidebar.success(f"✅ Loaded: **{selected_mol.upper()}**")

    # Load data
    with st.spinner(f"Loading data for {selected_mol.upper()}..."):
        result = run_processing_pipeline(selected_mol, file_discovery)

    if result['status'] == 'failed':
        st.error(f"❌ Failed to load data: {', '.join(result['errors'])}")
        st.stop()

    st.session_state.molecule_data = result
    st.session_state.pipeline_metadata = result['metadata']

    metadata        = result['metadata']
    cipla_baseline  = metadata['cipla_baseline']
    filter_stats    = metadata['filter_stats']
    consolidated_df = result['data']['consolidated']
    cipla_df_agg    = result['data']['cipla']

    # ── Top KPI row ─────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Files Loaded",     len(metadata['files_loaded']))
    with k2:
        st.metric("Raw Records",      f"{metadata['raw_record_count']:,}")
    with k3:
        st.metric("Filtered Records", f"{filter_stats['filtered_count']:,}",
                  delta=f"-{filter_stats['removal_percentage']:.1f}%")
    with k4:
        st.metric("Cipla Avg Qty",    f"{cipla_baseline['avg_qty']:,.0f}")
    with k5:
        st.metric("Cipla Avg Price",  f"₹{cipla_baseline['avg_price']:,.2f}")

    # ── Global filter bar ────────────────────────────────────────────────────
    with st.container():
        st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
        fc1, fc2, fc3, fc4 = st.columns(4)

        unique_dates = sorted(consolidated_df['yyyymm'].unique())
        with fc1:
            date_range = st.multiselect("📅 Period (yyyymm)", unique_dates,
                                        default=unique_dates, key="date_filter")
        with fc2:
            grade_specs    = sorted(consolidated_df['GRADE_SPEC'].unique())
            selected_grades = st.multiselect("🏷️ Grade/Spec", grade_specs,
                                             default=grade_specs, key="grade_filter")
        with fc3:
            uoms           = sorted(consolidated_df['uom'].unique())
            selected_uoms  = st.multiselect("📐 UOM", uoms,
                                            default=uoms, key="uom_filter")
        with fc4:
            sources        = sorted(consolidated_df['source'].unique())
            selected_sources = st.multiselect("🏭 Source", sources,
                                              default=sources, key="source_filter")

        st.markdown('</div>', unsafe_allow_html=True)

    # Apply filters
    filtered_df = consolidated_df[
        (consolidated_df['yyyymm'].isin(date_range)) &
        (consolidated_df['GRADE_SPEC'].isin(selected_grades)) &
        (consolidated_df['uom'].isin(selected_uoms)) &
        (consolidated_df['source'].isin(selected_sources))
    ]

    # ========================================================================
    # SECTION 1 – Cipla Wtd Average Price
    # ========================================================================
    st.markdown(
        '<div class="section-header">1. Cipla Wtd Average Price</div>',
        unsafe_allow_html=True
    )

    cipla_filtered = filtered_df[filtered_df['source'] == 'Cipla'].copy()

    s1_left, s1_right = st.columns([1, 3])

    with s1_left:
        # Weighted average price across the selected period
        if len(cipla_filtered) > 0:
            wtd_avg = _safe_wtd_avg(cipla_filtered['Sum_of_TOTAL_VALUE'],
                                    cipla_filtered['Sum_of_QTY'])
            total_qty   = cipla_filtered['Sum_of_QTY'].sum()
            total_value = cipla_filtered['Sum_of_TOTAL_VALUE'].sum()
        else:
            wtd_avg     = cipla_baseline['avg_price']
            total_qty   = cipla_baseline['avg_qty']
            total_value = 0.0

        st.markdown(f"""
        <div class="cipla-highlight">
          <div class="price-label">Weighted Average Price</div>
          <div class="main-price">₹{wtd_avg:,.2f}</div>
          <div class="price-label" style="margin-top:0.6rem;">per unit · {selected_mol.upper()}</div>
          <hr style="border-color:rgba(255,255,255,0.25);margin:0.9rem 0;">
          <div style="display:flex;justify-content:space-around;">
            <div>
              <div style="font-size:1rem;font-weight:700;">₹{total_value:,.0f}</div>
              <div style="font-size:0.7rem;color:#b3d1e8;">Total Value (₹)</div>
            </div>
            <div>
              <div style="font-size:1rem;font-weight:700;">{total_qty:,.0f}</div>
              <div style="font-size:0.7rem;color:#b3d1e8;">Total Qty</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with s1_right:
        cipla_trend = (
            cipla_filtered.groupby('yyyymm')
            .apply(lambda g: _safe_wtd_avg(g['Sum_of_TOTAL_VALUE'], g['Sum_of_QTY']))
            .reset_index(name='Wtd_Avg_Price')
        )
        if len(cipla_trend) > 0:
            fig_cipla = go.Figure()
            fig_cipla.add_trace(go.Scatter(
                x=cipla_trend['yyyymm'], y=cipla_trend['Wtd_Avg_Price'],
                mode='lines+markers',
                line=dict(color='#2e86c1', width=2.5),
                marker=dict(size=7, color='#1a3a5c'),
                fill='tozeroy', fillcolor='rgba(46,134,193,0.10)',
                name='Cipla Wtd Avg Price'
            ))
            fig_cipla.update_layout(
                title=None,
                xaxis_title="Period",
                yaxis_title="Wtd Avg Price (₹)",
                height=240,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='white', plot_bgcolor='#f9fbfe',
                font=dict(size=11)
            )
            st.plotly_chart(fig_cipla, use_container_width=True)
        else:
            st.info("No Cipla data available for the selected filters.")

    # Per-grade breakdown
    if len(cipla_filtered) > 0:
        grade_wtd = (
            cipla_filtered.groupby('GRADE_SPEC')
            .apply(lambda g: pd.Series({
                'Wtd Avg Price (₹)': _safe_wtd_avg(g['Sum_of_TOTAL_VALUE'], g['Sum_of_QTY']),
                'Total Qty':         g['Sum_of_QTY'].sum(),
                'Total Value (₹)':   g['Sum_of_TOTAL_VALUE'].sum()
            }))
            .reset_index()
        )
        grade_wtd['Wtd Avg Price (₹)'] = grade_wtd['Wtd Avg Price (₹)'].apply(lambda x: f"₹{x:,.2f}")
        grade_wtd['Total Value (₹)']   = grade_wtd['Total Value (₹)'].apply(lambda x: f"₹{x:,.2f}")
        grade_wtd['Total Qty']         = grade_wtd['Total Qty'].apply(lambda x: f"{x:,.0f}")
        st.caption("Cipla Wtd Avg Price by Grade/Spec")
        st.dataframe(grade_wtd, use_container_width=True, hide_index=True)

    # ========================================================================
    # SECTION 2 – Comparison with Benchmarks of Competitors
    # ========================================================================
    st.markdown(
        '<div class="section-header">2. Comparison with Benchmarks of Competitors for Select Period</div>',
        unsafe_allow_html=True
    )

    # Weighted avg price per source per period
    bench_data = (
        filtered_df.groupby(['yyyymm', 'source'])
        .apply(lambda g: pd.Series({
            'Wtd_Avg_Price': _safe_wtd_avg(g['Sum_of_TOTAL_VALUE'], g['Sum_of_QTY']),
            'Total_Qty':     g['Sum_of_QTY'].sum()
        }))
        .reset_index()
    )

    colour_map = {'Cipla': '#1a3a5c', 'Supplier': '#2e86c1', 'Buyer': '#27ae60'}

    b_left, b_right = st.columns(2)

    with b_left:
        fig_bar = px.bar(
            bench_data,
            x='yyyymm', y='Wtd_Avg_Price', color='source',
            barmode='group',
            color_discrete_map=colour_map,
            labels={'Wtd_Avg_Price': 'Wtd Avg Price (₹)', 'yyyymm': 'Period', 'source': 'Source'},
            height=360
        )
        fig_bar.update_layout(
            title=None, legend_title_text='Source',
            paper_bgcolor='white', plot_bgcolor='#f9fbfe',
            font=dict(size=11), margin=dict(l=10, r=10, t=10, b=10)
        )
        st.caption("Grouped bar – Wtd Avg Price by Period & Source")
        st.plotly_chart(fig_bar, use_container_width=True)

    with b_right:
        fig_line = px.line(
            bench_data,
            x='yyyymm', y='Wtd_Avg_Price', color='source',
            markers=True,
            color_discrete_map=colour_map,
            labels={'Wtd_Avg_Price': 'Wtd Avg Price (₹)', 'yyyymm': 'Period', 'source': 'Source'},
            height=360
        )
        fig_line.update_layout(
            title=None, legend_title_text='Source',
            paper_bgcolor='white', plot_bgcolor='#f9fbfe',
            font=dict(size=11), margin=dict(l=10, r=10, t=10, b=10)
        )
        st.caption("Trend line – Price movement over time")
        st.plotly_chart(fig_line, use_container_width=True)

    # Summary comparison table
    bench_summary = (
        filtered_df.groupby('source')
        .apply(lambda g: pd.Series({
            'Wtd Avg Price (₹)': _safe_wtd_avg(g['Sum_of_TOTAL_VALUE'], g['Sum_of_QTY']),
            'Total Qty':         g['Sum_of_QTY'].sum(),
            'Total Value (₹)':   g['Sum_of_TOTAL_VALUE'].sum(),
            'No. of Records':    len(g)
        }))
        .reset_index()
    )
    # Variance vs Cipla
    cipla_ref = bench_summary.loc[bench_summary['source'] == 'Cipla', 'Wtd Avg Price (₹)']
    cipla_ref_val = cipla_ref.values[0] if len(cipla_ref) > 0 else cipla_baseline['avg_price']
    bench_summary['vs Cipla (%)'] = bench_summary['Wtd Avg Price (₹)'].apply(
        lambda p: (f"{(p - cipla_ref_val) / cipla_ref_val * 100:+.2f}%"
                   if cipla_ref_val != 0 else "N/A")
    )
    bench_summary['Wtd Avg Price (₹)'] = bench_summary['Wtd Avg Price (₹)'].apply(lambda x: f"₹{x:,.2f}")
    bench_summary['Total Value (₹)']   = bench_summary['Total Value (₹)'].apply(lambda x: f"₹{x:,.2f}")
    bench_summary['Total Qty']         = bench_summary['Total Qty'].apply(lambda x: f"{x:,.0f}")

    st.dataframe(bench_summary, use_container_width=True, hide_index=True)

    # ========================================================================
    # SECTION 3 – Typical Bubble Chart Analysis
    # ========================================================================
    st.markdown(
        '<div class="section-header">3. Typical Bubble Chart Analysis</div>',
        unsafe_allow_html=True
    )

    bubble_data = (
        filtered_df.groupby(['yyyymm', 'source', 'GRADE_SPEC'])
        .apply(lambda g: pd.Series({
            'Wtd_Avg_Price': _safe_wtd_avg(g['Sum_of_TOTAL_VALUE'], g['Sum_of_QTY']),
            'Total_Qty':     g['Sum_of_QTY'].sum()
        }))
        .reset_index()
    )

    fig_bubble = px.scatter(
        bubble_data,
        x='Total_Qty', y='Wtd_Avg_Price',
        color='source', size='Total_Qty',
        symbol='GRADE_SPEC',
        hover_data=['yyyymm', 'GRADE_SPEC'],
        color_discrete_map=colour_map,
        labels={
            'Total_Qty':     'Total Volume (Qty)',
            'Wtd_Avg_Price': 'Wtd Avg Price (₹)',
            'source':        'Source',
            'GRADE_SPEC':    'Grade/Spec'
        },
        size_max=60,
        height=480
    )
    # Cipla baseline line
    fig_bubble.add_hline(
        y=cipla_baseline['avg_price'],
        line_dash='dash', line_color='#e74c3c',
        annotation_text=f"Cipla Baseline ₹{cipla_baseline['avg_price']:,.2f}",
        annotation_position='top right'
    )
    fig_bubble.update_layout(
        title=None,
        paper_bgcolor='white', plot_bgcolor='#f9fbfe',
        font=dict(size=11),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    st.plotly_chart(fig_bubble, use_container_width=True)
    st.caption(
        "Bubble size = purchase volume. "
        "Red dashed line = Cipla baseline price. "
        "Shape = Grade/Spec."
    )

    # ========================================================================
    # SECTION 4 – Volume and Price Data Tables
    # ========================================================================
    st.markdown(
        '<div class="section-header">4. Volume and Price Data Tables</div>',
        unsafe_allow_html=True
    )

    tab_supplier, tab_buyer, tab_cipla, tab_all = st.tabs([
        "🏭 Supplier", "🛒 Buyer", "🏢 Cipla", "📋 All Data"
    ])

    def _fmt_table(df_src):
        """Return a display-ready copy of a source-filtered DataFrame."""
        out = df_src[[
            'yyyymm', 'entity_name', 'uom', 'GRADE_SPEC',
            'Sum_of_QTY', 'Sum_of_TOTAL_VALUE', 'Avg_PRICE'
        ]].copy()
        out.columns = ['Period', 'Entity', 'UOM', 'Grade/Spec',
                       'Volume (Qty)', 'Total Value (₹)', 'Avg Price (₹)']
        out['Volume (Qty)']   = out['Volume (Qty)'].apply(lambda x: f"{x:,.0f}")
        out['Total Value (₹)'] = out['Total Value (₹)'].apply(lambda x: f"₹{x:,.2f}")
        out['Avg Price (₹)']  = out['Avg Price (₹)'].apply(lambda x: f"₹{x:,.2f}")
        return out.sort_values('Period', ascending=False)

    with tab_supplier:
        sup_df = _fmt_table(filtered_df[filtered_df['source'] == 'Supplier'])
        st.dataframe(sup_df, use_container_width=True, hide_index=True)
        csv = sup_df.to_csv(index=False)
        st.download_button("📥 Download Supplier Data", csv,
                           f"{selected_mol}_supplier.csv", "text/csv",
                           key="dl_supplier")

    with tab_buyer:
        buy_df = _fmt_table(filtered_df[filtered_df['source'] == 'Buyer'])
        st.dataframe(buy_df, use_container_width=True, hide_index=True)
        csv = buy_df.to_csv(index=False)
        st.download_button("📥 Download Buyer Data", csv,
                           f"{selected_mol}_buyer.csv", "text/csv",
                           key="dl_buyer")

    with tab_cipla:
        cip_df = _fmt_table(filtered_df[filtered_df['source'] == 'Cipla'])
        st.dataframe(cip_df, use_container_width=True, hide_index=True)
        csv = cip_df.to_csv(index=False)
        st.download_button("📥 Download Cipla Data", csv,
                           f"{selected_mol}_cipla.csv", "text/csv",
                           key="dl_cipla")

    with tab_all:
        all_df = _fmt_table(filtered_df)
        st.dataframe(all_df, use_container_width=True, hide_index=True)
        csv = all_df.to_csv(index=False)
        st.download_button("📥 Download All Data", csv,
                           f"{selected_mol}_all.csv", "text/csv",
                           key="dl_all")

# ============================================================================
# LANDING PAGE (no molecule selected)
# ============================================================================
else:
    st.markdown("""
    <div class="welcome-card">
      <h2 style="color:#1a3a5c;margin-top:0;">👋 Welcome to the Price Benchmarking Dashboard</h2>
      <p style="color:#555;font-size:0.95rem;">
        Select a molecule from the sidebar to explore Cipla's procurement intelligence across
        <strong>Suppliers</strong>, <strong>Buyers</strong>, and <strong>Cipla's own GRN data</strong>.
      </p>
      <hr>
      <p style="color:#1a3a5c;font-weight:700;margin-bottom:0.5rem;">The dashboard provides:</p>
    """, unsafe_allow_html=True)

    steps = [
        ("1", "Cipla wtd average price — see Cipla's weighted average procurement price with a trend chart."),
        ("2", "Comparison with benchmarks of competitors for select period — grouped bar & line chart vs Supplier and Buyer."),
        ("3", "Typical bubble chart analysis — volume vs price bubbles with the Cipla baseline reference line."),
        ("4", "Volume and price data tables — detailed per-source tables with CSV download."),
    ]
    for num, text in steps:
        st.markdown(f"""
        <div class="step-item">
          <div class="step-num">{num}</div>
          <div class="step-text">{text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Molecule cards
    st.markdown(
        '<div class="section-header">Available Molecules</div>',
        unsafe_allow_html=True
    )
    mol_cols = st.columns(min(len(available_molecules), 4))
    for idx, (mol_name, mol_info) in enumerate(available_molecules.items()):
        with mol_cols[idx % 4]:
            st.markdown(f"""
            <div class="kpi-card accent" style="text-align:center;padding:1.2rem;">
              <div class="kpi-label">Molecule</div>
              <div class="kpi-value" style="font-size:1.1rem;">{mol_name.upper()}</div>
              <div style="font-size:0.72rem;color:#666;margin-top:0.3rem;">{mol_info['description']}</div>
              <div style="font-size:0.72rem;color:#888;margin-top:0.2rem;">
                {mol_info['file_count']} file(s) &nbsp;|&nbsp;
                {'✓ Cipla' if mol_info['cipla_available'] else '✗ No Cipla'}
              </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Load {mol_name.upper()}", key=f"land_{mol_name}"):
                st.session_state.selected_molecule = mol_name
                st.rerun()
