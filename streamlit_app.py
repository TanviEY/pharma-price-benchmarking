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
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .title-main {
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
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
# HEADER
# ============================================================================
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<h1 class="title-main">💊 Pharmaceutical Price Benchmarking</h1>', unsafe_allow_html=True)
with col2:
    st.write("")
    st.write("")
    current_date = datetime.now().strftime("%Y-%m-%d")
    st.caption(f"Last Updated: {current_date}")

st.markdown("---")

# ============================================================================
# SIDEBAR - MOLECULE SELECTION
# ============================================================================
st.sidebar.title("🔍 Molecule Selection")

available_molecules = file_discovery.get_available_molecules()

# Search box with fuzzy matching
search_input = st.sidebar.text_input(
    "Search molecule:",
    placeholder="e.g., azithromycoon, cipro, amoxy...",
    help="Type a molecule name. Supports typos and aliases!"
)

matches = []
if search_input.strip():
    matches = fuzzy_matcher.match_molecule_input(search_input)

# Display search results
if matches:
    st.sidebar.subheader("🎯 Suggestions")
    
    for mol_name, confidence in matches:
        if mol_name in available_molecules:
            col1, col2, col3 = st.sidebar.columns([2, 1, 1])
            
            with col1:
                st.write(f"**{mol_name.upper()}**")
            with col2:
                st.metric("Match", f"{confidence}%")
            with col3:
                if st.button("Select", key=f"btn_{mol_name}"):
                    st.session_state.selected_molecule = mol_name
                    st.rerun()
elif search_input.strip():
    st.sidebar.warning("❌ No matches found")
    st.sidebar.info("💡 Check spelling or select from available molecules below")

# Display available molecules
st.sidebar.subheader("📋 Available Molecules")

for mol_name, mol_info in available_molecules.items():
    with st.sidebar.expander(f"✓ {mol_name.upper()}", expanded=False):
        st.write(f"**Description:** {mol_info['description']}")
        st.write(f"**Files:** {mol_info['file_count']}")
        st.write(f"**Cipla Data:** {'✓ Yes' if mol_info['cipla_available'] else '✗ No'}")
        
        if st.button("Load", key=f"load_{mol_name}"):
            st.session_state.selected_molecule = mol_name
            st.rerun()

# ============================================================================
# MAIN CONTENT
# ============================================================================

if st.session_state.selected_molecule:
    selected_mol = st.session_state.selected_molecule
    
    st.sidebar.success(f"✓ Selected: **{selected_mol.upper()}**")
    
    # Load data with spinner
    with st.spinner(f"Loading data for {selected_mol.upper()}..."):
        result = run_processing_pipeline(selected_mol, file_discovery)
    
    if result['status'] == 'failed':
        st.error(f"❌ Failed to load data: {', '.join(result['errors'])}")
        st.stop()
    
    # Store in session state
    st.session_state.molecule_data = result
    st.session_state.pipeline_metadata = result['metadata']
    
    # ========================================================================
    # DATA SUMMARY
    # ========================================================================
    st.header(f"{selected_mol.upper()} - Data Summary")
    
    metadata = result['metadata']
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Files Loaded",
            len(metadata['files_loaded']),
            help="Number of source files"
        )
    
    with col2:
        st.metric(
            "Raw Records",
            f"{metadata['raw_record_count']:,}",
            help="Total records before filtering"
        )
    
    with col3:
        filter_stats = metadata['filter_stats']
        st.metric(
            "Filtered Records",
            f"{filter_stats['filtered_count']:,}",
            delta=f"-{filter_stats['removal_percentage']:.1f}%",
            help="Records after outlier removal"
        )
    
    with col4:
        cipla_baseline = metadata['cipla_baseline']
        st.metric(
            "Cipla Baseline Qty",
            f"{cipla_baseline['avg_qty']:,.0f}",
            help="Average Cipla quantity"
        )
    
    with col5:
        st.metric(
            "Cipla Baseline Price",
            f"₹{cipla_baseline['avg_price']:,.2f}",
            help="Average Cipla price"
        )
    
    st.markdown("---")
    
    # ========================================================================
    # FILTERS
    # ========================================================================
    st.header("🔧 Filters")
    
    # Get consolidated data for filter options
    consolidated_df = result['data']['consolidated']
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Date range filter
    with col1:
        unique_dates = sorted(consolidated_df['yyyymm'].unique())
        date_range = st.multiselect(
            "Date Range (yyyymm):",
            options=unique_dates,
            default=unique_dates,
            key="date_filter"
        )
    
    # Grade/Spec filter
    with col2:
        grade_specs = sorted(consolidated_df['GRADE_SPEC'].unique())
        selected_grades = st.multiselect(
            "Grade/Spec:",
            options=grade_specs,
            default=grade_specs,
            key="grade_filter"
        )
    
    # UOM filter
    with col3:
        uoms = sorted(consolidated_df['uom'].unique())
        selected_uoms = st.multiselect(
            "Unit of Measure:",
            options=uoms,
            default=uoms,
            key="uom_filter"
        )
    
    # Source filter
    with col4:
        sources = sorted(consolidated_df['source'].unique())
        selected_sources = st.multiselect(
            "Data Source:",
            options=sources,
            default=sources,
            key="source_filter"
        )
    
    # Apply filters
    filters = {
        'date_range': date_range,
        'grade_specs': selected_grades,
        'uoms': selected_uoms,
        'sources': selected_sources
    }
    
    # Filter consolidated data
    filtered_df = consolidated_df[
        (consolidated_df['yyyymm'].isin(date_range)) &
        (consolidated_df['GRADE_SPEC'].isin(selected_grades)) &
        (consolidated_df['uom'].isin(selected_uoms)) &
        (consolidated_df['source'].isin(selected_sources))
    ]
    
    st.markdown("---")
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    st.header("📊 Visualizations")
    
    # TAB 1: Price Trends
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Price Trends",
        "📦 Price Distribution",
        "🔗 Volume vs Price",
        "📊 Price Variance",
        "📋 Detailed Data"
    ])
    
    with tab1:
        st.subheader("Price Trend Over Time")
        
        # Prepare data for line chart
        trend_data = filtered_df.groupby(['yyyymm', 'source']).agg({
            'Avg_PRICE': 'mean'
        }).reset_index()
        
        fig = px.line(
            trend_data,
            x='yyyymm',
            y='Avg_PRICE',
            color='source',
            markers=True,
            title="Average Price Trend by Source",
            labels={'Avg_PRICE': 'Average Price (₹)', 'yyyymm': 'Month'},
            height=500
        )
        
        fig.update_yaxes(title_text="Average Price (₹)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        
        for source in selected_sources:
            source_data = filtered_df[filtered_df['source'] == source]
            
            if len(source_data) > 0:
                avg_price = source_data['Avg_PRICE'].mean()
                max_price = source_data['Avg_PRICE'].max()
                min_price = source_data['Avg_PRICE'].min()
                
                if source == selected_sources[0]:
                    with col1:
                        st.metric(f"{source} Avg Price", f"₹{avg_price:,.2f}")
                        st.caption(f"Min: ₹{min_price:,.2f} | Max: ₹{max_price:,.2f}")
                elif source == selected_sources[1] if len(selected_sources) > 1 else False:
                    with col2:
                        st.metric(f"{source} Avg Price", f"₹{avg_price:,.2f}")
                        st.caption(f"Min: ₹{min_price:,.2f} | Max: ₹{max_price:,.2f}")
                else:
                    with col3:
                        st.metric(f"{source} Avg Price", f"₹{avg_price:,.2f}")
                        st.caption(f"Min: ₹{min_price:,.2f} | Max: ₹{max_price:,.2f}")
    
    with tab2:
        st.subheader("Price Distribution by Source")
        
        fig = px.box(
            filtered_df,
            x='source',
            y='Avg_PRICE',
            color='source',
            title="Price Distribution (Box Plot)",
            labels={'Avg_PRICE': 'Average Price (₹)', 'source': 'Source'},
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional box plot by grade/spec
        if len(selected_grades) > 1:
            fig2 = px.box(
                filtered_df,
                x='GRADE_SPEC',
                y='Avg_PRICE',
                color='source',
                title="Price Distribution by Grade/Spec",
                labels={'Avg_PRICE': 'Average Price (₹)', 'GRADE_SPEC': 'Grade/Spec'},
                height=500
            )
            
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.subheader("Volume vs Price Analysis")
        
        scatter_data = filtered_df.groupby(['yyyymm', 'source']).agg({
            'Sum_of_QTY': 'sum',
            'Avg_PRICE': 'mean'
        }).reset_index()
        
        fig = px.scatter(
            scatter_data,
            x='Sum_of_QTY',
            y='Avg_PRICE',
            color='source',
            size='Sum_of_QTY',
            hover_data=['yyyymm'],
            title="Volume vs Price Relationship",
            labels={'Sum_of_QTY': 'Total Quantity', 'Avg_PRICE': 'Average Price (₹)'},
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Price Variance Analysis")
        
        # Calculate variance from Cipla baseline
        cipla_baseline_price = metadata['cipla_baseline']['avg_price']
        
        variance_data = filtered_df[filtered_df['source'] != 'Cipla'].copy()
        variance_data['price_variance_pct'] = (
            (variance_data['Avg_PRICE'] - cipla_baseline_price) / cipla_baseline_price * 100
        )
        
        variance_agg = variance_data.groupby(['source']).agg({
            'price_variance_pct': 'mean',
            'Avg_PRICE': 'mean'
        }).reset_index()
        
        fig = px.bar(
            variance_agg,
            x='source',
            y='price_variance_pct',
            color='price_variance_pct',
            title=f"Price Variance from Cipla Baseline (₹{cipla_baseline_price:,.2f})",
            labels={'price_variance_pct': 'Variance (%)', 'source': 'Source'},
            height=500,
            color_continuous_scale='RdYlGn_r'
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Baseline")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Variance table
        st.write("**Price Variance Summary:**")
        variance_summary = variance_agg.copy()
        variance_summary['Variance (%)'] = variance_summary['price_variance_pct'].apply(
            lambda x: f"{x:+.2f}%"
        )
        variance_summary['Avg Price'] = variance_summary['Avg_PRICE'].apply(
            lambda x: f"₹{x:,.2f}"
        )
        
        st.dataframe(
            variance_summary[['source', 'Variance (%)', 'Avg Price']],
            use_container_width=True,
            hide_index=True
        )
    
    with tab5:
        st.subheader("Detailed Data Table")
        
        # Prepare display dataframe
        display_df = filtered_df[[
            'yyyymm', 'source', 'entity_name', 'uom', 'GRADE_SPEC',
            'Sum_of_QTY', 'Sum_of_TOTAL_VALUE', 'Avg_PRICE'
        ]].copy()
        
        display_df.columns = [
            'Month', 'Source', 'Entity', 'UOM', 'Grade/Spec',
            'Qty', 'Total Value (₹)', 'Avg Price (₹)'
        ]
        
        # Format numeric columns
        display_df['Qty'] = display_df['Qty'].apply(lambda x: f"{x:,.0f}")
        display_df['Total Value (₹)'] = display_df['Total Value (₹)'].apply(lambda x: f"₹{x:,.2f}")
        display_df['Avg Price (₹)'] = display_df['Avg Price (₹)'].apply(lambda x: f"₹{x:,.2f}")
        
        # Sort by month descending
        display_df = display_df.sort_values('Month', ascending=False)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"{selected_mol}_benchmarking_data.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    
    # ========================================================================
    # DETAILED STATISTICS
    # ========================================================================
    st.header("📊 Detailed Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("By Grade/Spec")
        grade_stats = filtered_df.groupby('GRADE_SPEC').agg({
            'Sum_of_QTY': 'sum',
            'Sum_of_TOTAL_VALUE': 'sum',
            'Avg_PRICE': 'mean'
        }).reset_index()
        
        grade_stats['Avg_PRICE'] = grade_stats['Avg_PRICE'].apply(lambda x: f"₹{x:,.2f}")
        grade_stats['Sum_of_TOTAL_VALUE'] = grade_stats['Sum_of_TOTAL_VALUE'].apply(lambda x: f"₹{x:,.2f}")
        grade_stats.columns = ['Grade/Spec', 'Total Qty', 'Total Value', 'Avg Price']
        
        st.dataframe(grade_stats, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("By UOM")
        uom_stats = filtered_df.groupby('uom').agg({
            'Sum_of_QTY': 'sum',
            'Sum_of_TOTAL_VALUE': 'sum',
            'Avg_PRICE': 'mean'
        }).reset_index()
        
        uom_stats['Avg_PRICE'] = uom_stats['Avg_PRICE'].apply(lambda x: f"₹{x:,.2f}")
        uom_stats['Sum_of_TOTAL_VALUE'] = uom_stats['Sum_of_TOTAL_VALUE'].apply(lambda x: f"₹{x:,.2f}")
        uom_stats.columns = ['UOM', 'Total Qty', 'Total Value', 'Avg Price']
        
        st.dataframe(uom_stats, use_container_width=True, hide_index=True)

else:
    # Landing page
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 👋 Welcome to Price Benchmarking Dashboard
        
        This application helps you analyze and benchmark pharmaceutical prices across:
        - **Suppliers**: Direct supplier pricing data
        - **Buyers**: Importer/buyer pricing data
        - **Cipla**: Internal reference pricing
        
        ### How to use:
        1. **Search for a molecule** using the search box on the left (supports typos!)
        2. **View available molecules** and their data status
        3. **Apply filters** for specific date ranges, grades, and sources
        4. **Analyze visualizations** for price trends and distributions
        5. **Download data** for further analysis
        
        ### Supported molecules:
        """)
        
        for mol_name in available_molecules:
            st.write(f"✓ **{mol_name.upper()}**")
    
    with col2:
        st.info("""
        ### 💡 Tips:
        - Fuzzy matching helps find molecules even with typos
        - Use filters to focus on specific time periods
        - Click charts for interactive options
        - Download data for external analysis
        """)
