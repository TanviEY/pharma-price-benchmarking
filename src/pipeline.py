# pipeline.py
from utils import discover_molecule_files, load_molecule_files, load_cipla_grn, calculate_cipla_baseline, apply_outlier_filters, aggregate_by_supplier, aggregate_by_buyer, aggregate_cipla
from config import molecule_mapping

def run_processing_pipeline(molecule_name):
    """
    Universal pipeline for any molecule
    """
    # Step 1: Discover files
    files = discover_molecule_files(molecule_name)
    
    # Step 2: Load data
    molecule_df = load_molecule_files(files, molecule_name)
    cipla_df = load_cipla_grn(molecule_mapping[molecule_name]['cipla_api_filter'])
    
    # Step 3: Calculate baselines
    cipla_baseline = calculate_cipla_baseline(cipla_df)
    
    # Step 4: Filter outliers
    molecule_df_filtered = apply_outlier_filters(molecule_df, cipla_baseline)
    
    # Step 5: Aggregate
    supplier_agg = aggregate_by_supplier(molecule_df_filtered)
    buyer_agg = aggregate_by_buyer(molecule_df_filtered)
    cipla_agg = aggregate_cipla(cipla_df)
    
    # Step 6: Save outputs
    supplier_agg.to_csv(f"data/processed/{molecule_name}_supplier.csv", index=False)
    buyer_agg.to_csv(f"data/processed/{molecule_name}_buyer.csv", index=False)
    cipla_agg.to_csv(f"data/processed/cipla_{molecule_name}.csv", index=False)
    
    return {
        "supplier": supplier_agg,
        "buyer": buyer_agg,
        "cipla": cipla_agg,
        "baseline": cipla_baseline
    }