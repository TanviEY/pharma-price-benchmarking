# src/utils.py
import pandas as pd
import numpy as np
from typing import List, Dict

class Utils:
    """Utility functions"""
    
    @staticmethod
    def format_currency(value: float) -> str:
        """Format value as currency"""
        return f"₹{value:,.2f}"
    
    @staticmethod
    def format_percentage(value: float) -> str:
        """Format as percentage"""
        return f"{value:.2f}%"
    
    @staticmethod
    def calculate_price_variance(actual_price: float, baseline_price: float) -> float:
        """Calculate price variance from baseline"""
        if baseline_price == 0:
            return 0
        return ((actual_price - baseline_price) / baseline_price) * 100
    
    @staticmethod
    def get_grade_spec_options(df: pd.DataFrame) -> List[str]:
        """Get unique grade/spec values"""
        return sorted(df['GRADE_SPEC'].unique().tolist())
    
    @staticmethod
    def get_uom_options(df: pd.DataFrame) -> List[str]:
        """Get unique UOM values"""
        return sorted(df['uom'].unique().tolist())
    
    @staticmethod
    def get_date_range(df: pd.DataFrame) -> tuple:
        """Get min and max dates"""
        dates = pd.to_datetime(df['yyyymm'], format='%Y%m')
        return dates.min(), dates.max()
    
    @staticmethod
    def filter_dataframe(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply filters to dataframe"""
        filtered = df.copy()
        
        if 'date_range' in filters and filters['date_range']:
            min_date, max_date = filters['date_range']
            filtered = filtered[
                (filtered['yyyymm'] >= min_date.strftime('%Y%m')) &
                (filtered['yyyymm'] <= max_date.strftime('%Y%m'))
            ]
        
        if 'grade_specs' in filters and filters['grade_specs']:
            filtered = filtered[filtered['GRADE_SPEC'].isin(filters['grade_specs'])]
        
        if 'uoms' in filters and filters['uoms']:
            filtered = filtered[filtered['uom'].isin(filters['uoms'])]
        
        if 'sources' in filters and filters['sources']:
            filtered = filtered[filtered['source'].isin(filters['sources'])]
        
        return filtered