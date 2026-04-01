# src/data_processor.py
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from datetime import datetime

class DataProcessor:
    """Process and aggregate data"""
    
    @staticmethod
    def extract_grade_spec(item_text: str) -> str:
        """
        Extract grade/spec from ITEM description
        Hierarchy: IP < EP < USP
        Default: USP if none found
        """
        if not isinstance(item_text, str):
            return 'USP'
        
        item_lower = item_text.lower()
        found_specs = []
        
        if 'ip' in item_lower:
            found_specs.append('IP')
        if 'ep' in item_lower:
            found_specs.append('EP')
        if 'usp' in item_lower:
            found_specs.append('USP')
        
        # Apply hierarchy
        if 'USP' in found_specs:
            return 'USP'
        elif 'EP' in found_specs:
            return 'EP'
        elif 'IP' in found_specs:
            return 'IP'
        else:
            return 'USP'  # Default
    
    @staticmethod
    def extract_yyyymm(date_col) -> str:
        """Extract yyyymm from date"""
        try:
            if isinstance(date_col, str):
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
    def prepare_azithromycin_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare azithromycin data with grade/spec extraction"""
        df = df.copy()
        
        # Extract date
        df['yyyymm'] = df['BE_DATE'].apply(DataProcessor.extract_yyyymm)
        
        # Extract grade/spec
        df['GRADE_SPEC'] = df['ITEM'].apply(DataProcessor.extract_grade_spec)
        
        # Ensure QTY is numeric
        df['QTY'] = pd.to_numeric(df['QTY'], errors='coerce')
        df['TOTAL_VALUE'] = pd.to_numeric(df['TOTAL_VALUE'], errors='coerce')
        
        # Remove rows with missing critical values
        df = df.dropna(subset=['yyyymm', 'QTY', 'TOTAL_VALUE'])
        
        return df
    
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
    def aggregate_supplier(df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data by Supplier"""
        agg_df = df.groupby(['Supp_Name', 'yyyymm', 'UQC', 'GRADE_SPEC']).agg({
            'QTY': 'sum',
            'TOTAL_VALUE': 'sum'
        }).reset_index()
        
        agg_df.rename(columns={
            'Supp_Name': 'supplier',
            'UQC': 'uom',
            'QTY': 'Sum_of_QTY',
            'TOTAL_VALUE': 'Sum_of_TOTAL_VALUE'
        }, inplace=True)
        
        agg_df['Avg_PRICE'] = agg_df['Sum_of_TOTAL_VALUE'] / agg_df['Sum_of_QTY']
        agg_df['source'] = 'Supplier'
        
        return agg_df[['supplier', 'yyyymm', 'uom', 'GRADE_SPEC', 'Sum_of_QTY', 'Sum_of_TOTAL_VALUE', 'Avg_PRICE', 'source']]
    
    @staticmethod
    def aggregate_buyer(df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data by Buyer (Importer)"""
        agg_df = df.groupby(['IMPORTER', 'yyyymm', 'UQC', 'GRADE_SPEC']).agg({
            'QTY': 'sum',
            'TOTAL_VALUE': 'sum'
        }).reset_index()
        
        agg_df.rename(columns={
            'IMPORTER': 'buyer',
            'UQC': 'uom',
            'QTY': 'Sum_of_QTY',
            'TOTAL_VALUE': 'Sum_of_TOTAL_VALUE'
        }, inplace=True)
        
        agg_df['Avg_PRICE'] = agg_df['Sum_of_TOTAL_VALUE'] / agg_df['Sum_of_QTY']
        agg_df['source'] = 'Buyer'
        
        return agg_df[['buyer', 'yyyymm', 'uom', 'GRADE_SPEC', 'Sum_of_QTY', 'Sum_of_TOTAL_VALUE', 'Avg_PRICE', 'source']]
    
    @staticmethod
    def aggregate_cipla(cipla_df: pd.DataFrame) -> pd.DataFrame:
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
        agg_df['api'] = 'AZITHROMYCIN'
        
        return agg_df[['api', 'yyyymm', 'uom', 'GRADE_SPEC', 'Sum_of_QTY', 'Sum_of_TOTAL_VALUE', 'Avg_PRICE', 'source']]