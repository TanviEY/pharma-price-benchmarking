# pipeline.py
import os
import pandas as pd
from pathlib import Path
from src.file_discovery import FileDiscovery 
from src.data_loader import DataLoader 
from src.data_processor import DataProcessor
from src.settings import MOLECULE_MAPPING


def run_processing_pipeline(molecule_name, file_discovery):
    """
    Universal pipeline for any molecule
    """
    try:
        # Step 1: Discover files
        mol_files = file_discovery.discover_molecule_files(molecule_name)
        cipla_file = file_discovery.discover_cipla_file()

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
        molecule_df = DataLoader.load_multiple_files(mol_files)
        api_filter = MOLECULE_MAPPING['molecules'][molecule_name]['cipla_api_filter']
        cipla_df = DataLoader.load_cipla_grn(cipla_file, api_filter)

        raw_record_count = len(molecule_df)

        # Step 3: Prepare data
        molecule_df = DataProcessor.prepare_azithromycin_data(molecule_df)
        cipla_df = DataProcessor.prepare_cipla_data(cipla_df)

        # Step 4: Calculate baselines
        cipla_baseline = DataProcessor.calculate_cipla_baseline(cipla_df)

        # Step 5: Filter outliers (returns tuple of (df, stats))
        molecule_df_filtered, filter_stats = DataProcessor.apply_outlier_filters(molecule_df, cipla_baseline)

        # Step 6: Aggregate
        supplier_agg = DataProcessor.aggregate_supplier(molecule_df_filtered)
        buyer_agg = DataProcessor.aggregate_buyer(molecule_df_filtered)
        cipla_agg = DataProcessor.aggregate_cipla(cipla_df, molecule_name)

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
                'consolidated': consolidated
            }
        }

    except Exception as e:
        return {
            'status': 'failed',
            'errors': [str(e)]
        }