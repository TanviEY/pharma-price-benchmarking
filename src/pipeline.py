# pipeline.py
import pandas as pd
from pathlib import Path
from src.backend import (
    discover_molecule_files,
    discover_cipla_file,
    load_multiple_files,
    load_cipla_grn,
    prepare_molecule_data,
    prepare_cipla_data,
    calculate_cipla_baseline,
    apply_outlier_filters,
    aggregate_supplier,
    aggregate_buyer,
    aggregate_cipla,
)
from src.settings import MOLECULE_MAPPING

_DATA_DIR = "data/raw"


def run_processing_pipeline(molecule_name: str, data_dir: str = _DATA_DIR) -> dict:
    """
    Universal pipeline for any molecule.
    """
    try:
        # Step 1: Discover files
        mol_files = discover_molecule_files(data_dir, MOLECULE_MAPPING, molecule_name)
        cipla_file = discover_cipla_file(data_dir)

        if not mol_files:
            return {
                'status': 'failed',
                'errors': [f'No data files found for molecule: {molecule_name}'],
            }
        if not cipla_file:
            return {
                'status': 'failed',
                'errors': ['Cipla GRN file not found'],
            }

        # Step 2: Load data
        molecule_df = load_multiple_files(mol_files)
        api_filter = MOLECULE_MAPPING['molecules'][molecule_name]['cipla_api_filter']
        cipla_df = load_cipla_grn(cipla_file, api_filter)

        raw_record_count = len(molecule_df)

        # Step 3: Prepare data
        molecule_df = prepare_molecule_data(molecule_df)
        cipla_df = prepare_cipla_data(cipla_df)

        # Step 4: Calculate baselines
        cipla_baseline = calculate_cipla_baseline(cipla_df)

        # Step 5: Filter outliers (returns tuple of (df, stats))
        molecule_df_filtered, filter_stats = apply_outlier_filters(molecule_df, cipla_baseline)

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
            ignore_index=True,
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
                'cipla_baseline': cipla_baseline,
            },
            'data': {
                'supplier': supplier_agg,
                'buyer': buyer_agg,
                'cipla': cipla_agg,
                'consolidated': consolidated,
            },
        }

    except Exception as e:
        return {
            'status': 'failed',
            'errors': [str(e)],
        }