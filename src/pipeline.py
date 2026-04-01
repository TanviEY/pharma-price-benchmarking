# pipeline.py
from src.file_discovery import FileDiscovery 
from src.data_loader import DataLoader 
from src.data_processor import DataProcessor
from src.settings import MOLECULE_MAPPING

dl = DataLoader()
dp = DataProcessor()

def run_processing_pipeline(molecule_name, file_discovery):
    """
    Universal pipeline for any molecule
    """
    # Step 1: Discover files
    files = file_discovery.discover_molecule_files(molecule_name)
    
    # Step 2: Load data
    molecule_df = dl.load_molecule_file(files, molecule_name)
    cipla_df = dl.load_cipla_grn(MOLECULE_MAPPING[molecule_name]['cipla_api_filter'])    
    
    # Step 3: Calculate baselines
    cipla_baseline = dp.calculate_cipla_baseline(cipla_df)
    
    # Step 4: Filter outliers
    molecule_df_filtered = dp.apply_outlier_filters(molecule_df, cipla_baseline)
    
    # Step 5: Aggregate
    supplier_agg = dp.aggregate_supplier(molecule_df_filtered)
    buyer_agg = dp.aggregate_buyer(molecule_df_filtered)
    cipla_agg = dp.aggregate_cipla(cipla_df)
    
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