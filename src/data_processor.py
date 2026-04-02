# src/data_processor.py
import pandas as pd
from typing import Tuple, Dict

class DataProcessor:
    """Process and aggregate data"""
    
    @staticmethod
    def extract_grade_spec(item_text: str) -> str:
        """
        Extract grade/spec from ITEM description.
        Hierarchy (highest wins): USP > EP > IP. Defaults to USP if none found.
        """
        if not isinstance(item_text, str):
            return 'USP'
        item_lower = item_text.lower()
        if 'usp' in item_lower:
            return 'USP'
        if 'ep' in item_lower:
            return 'EP'
        if 'ip' in item_lower:
            return 'IP'
        return 'USP'
    
    @staticmethod
    def extract_yyyymm(date_col) -> str:
        """Extract yyyymm from date"""
        try:
            if isinstance(date_col, (int, float)):
                date_obj = pd.to_datetime(str(int(date_col)), format='%Y%m%d')
            elif isinstance(date_col, str):
                date_obj = pd.to_datetime(date_col)
            else:
                date_obj = date_col
            return date_obj.strftime('%Y%m')
        except:
            return None
    
    @staticmethod
    def apply_outlier_filters(df: pd.DataFrame, cipla_baseline: Dict) -> pd.DataFrame:
        """
        Apply outlier filters based on Cipla baseline
        1. QTY >= 10% of Cipla Avg Qty
        2. TOTAL_VALUE within Cipla Price ± 30%
        """
        original_count = len(df)
        
        # Filter 1: Quantity threshold
        min_qty = cipla_baseline['avg_qty'] * 0.10
        df = df[df['QTY'] >= min_qty]
        
        # Filter 2: Price variance
        cipla_price = cipla_baseline['avg_price']
        price_lower = cipla_price * 0.70  # -30%
        price_upper = cipla_price * 1.30  # +30%
        
        # Calculate per-unit price
        df['unit_price'] = df['TOTAL_VALUE'] / df['QTY']
        df = df[(df['unit_price'] >= price_lower) & (df['unit_price'] <= price_upper)]
        
        filtered_count = len(df)
        removed = original_count - filtered_count
        
        return df, {
            'original_count': original_count,
            'filtered_count': filtered_count,
            'removed_count': removed,
            'removal_percentage': (removed / original_count * 100) if original_count > 0 else 0
        }
    
    @staticmethod
    def prepare_molecule_data(df: pd.DataFrame) -> pd.DataFrame:
        """Prepare molecule EXIM data - works for any molecule file"""
        df = df.copy()

        # Find date column
        date_col = None
        for col in ['BE_DATE', 'be_date', 'DATE', 'date', 'BILL_DATE']:
            if col in df.columns:
                date_col = col
                break
        if date_col is None:
            raise ValueError(f"No date column found. Available columns: {list(df.columns)}")
        df['yyyymm'] = df[date_col].apply(DataProcessor.extract_yyyymm)

        # Find item/description column for grade spec
        item_col = None
        for col in ['ITEM', 'item', 'ITEM_DESC', 'DESCRIPTION', 'PRODUCT']:
            if col in df.columns:
                item_col = col
                break
        if item_col:
            df['GRADE_SPEC'] = df[item_col].apply(DataProcessor.extract_grade_spec)
        else:
            df['GRADE_SPEC'] = 'USP'

        # Ensure QTY is numeric
        df['QTY'] = pd.to_numeric(df['QTY'], errors='coerce')
        df['TOTAL_VALUE'] = pd.to_numeric(df['TOTAL_VALUE'], errors='coerce')

        # Remove rows with missing critical values
        df = df.dropna(subset=['yyyymm', 'QTY', 'TOTAL_VALUE'])

        return df

    # Keep old name as alias for backward compatibility
    prepare_azithromycin_data = prepare_molecule_data
    
    @staticmethod
    def prepare_cipla_data(df: pd.DataFrame) -> pd.DataFrame:
        """Prepare Cipla GRN data"""
        df = df.copy()
        
        # Extract date
        df['yyyymm'] = df['posting_date_in_the_document'].apply(DataProcessor.extract_yyyymm)
        
        # Ensure numeric columns
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['actual_spend_inr'] = pd.to_numeric(df['actual_spend_inr'], errors='coerce')
        
        # Remove rows with missing critical values
        df = df.dropna(subset=['yyyymm', 'quantity', 'actual_spend_inr'])
        
        return df
    
    @staticmethod
    def calculate_cipla_baseline(cipla_df: pd.DataFrame) -> Dict:
        """Calculate baseline metrics from Cipla data"""
        return {
            'avg_qty': cipla_df['quantity'].mean(),
            'avg_price': (cipla_df['actual_spend_inr'] / cipla_df['quantity']).mean(),
            'min_qty': cipla_df['quantity'].min(),
            'max_qty': cipla_df['quantity'].max(),
            'total_records': len(cipla_df)
        }
    
    @staticmethod
    def _aggregate_entity(df: pd.DataFrame, entity_col: str, entity_alias: str, source: str) -> pd.DataFrame:
        """Shared aggregation logic for supplier and buyer views."""
        agg_df = df.groupby([entity_col, 'yyyymm', 'UQC', 'GRADE_SPEC']).agg(
            Sum_of_QTY=('QTY', 'sum'),
            Sum_of_TOTAL_VALUE=('TOTAL_VALUE', 'sum')
        ).reset_index()
        agg_df.rename(columns={entity_col: entity_alias, 'UQC': 'uom'}, inplace=True)
        agg_df['Avg_PRICE'] = agg_df['Sum_of_TOTAL_VALUE'] / agg_df['Sum_of_QTY']
        agg_df['source'] = source
        return agg_df[[entity_alias, 'yyyymm', 'uom', 'GRADE_SPEC', 'Sum_of_QTY', 'Sum_of_TOTAL_VALUE', 'Avg_PRICE', 'source']]

    @staticmethod
    def aggregate_supplier(df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data by Supplier"""
        return DataProcessor._aggregate_entity(df, 'Supp_Name', 'supplier', 'Supplier')

    @staticmethod
    def aggregate_buyer(df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data by Buyer (Importer)"""
        return DataProcessor._aggregate_entity(df, 'IMPORTER', 'buyer', 'Buyer')
    
    @staticmethod
    def aggregate_cipla(cipla_df: pd.DataFrame, molecule_name: str = 'unknown') -> pd.DataFrame:
        """Aggregate Cipla data"""
        agg_df = cipla_df.groupby(['yyyymm', 'base_unit_of_measure', 'grade_spec']).agg({
            'quantity': 'sum',
            'actual_spend_inr': 'sum'
        }).reset_index()
        
        agg_df.rename(columns={
            'base_unit_of_measure': 'uom',
            'grade_spec': 'GRADE_SPEC',
            'quantity': 'Sum_of_QTY',
            'actual_spend_inr': 'Sum_of_TOTAL_VALUE'
        }, inplace=True)
        
        agg_df['Avg_PRICE'] = agg_df['Sum_of_TOTAL_VALUE'] / agg_df['Sum_of_QTY']
        agg_df['source'] = 'Cipla'
        agg_df['api'] = molecule_name.upper()
        
        return agg_df[['api', 'yyyymm', 'uom', 'GRADE_SPEC', 'Sum_of_QTY', 'Sum_of_TOTAL_VALUE', 'Avg_PRICE', 'source']]