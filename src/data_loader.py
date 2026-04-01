# src/data_loader.py
import pandas as pd
from typing import Optional
import warnings

warnings.filterwarnings('ignore')

class DataLoader:
    """Load data from Excel files"""

    @staticmethod
    def load_cipla_grn(file_path: str, api_filter: Optional[str] = None) -> pd.DataFrame:
        """Load Cipla GRN data and filter by API"""
        try:
            df = pd.read_excel(file_path)
            if api_filter and 'api_family' in df.columns:
                df = df[df['api_family'].str.contains(api_filter, case=False, na=False)]
            return df
        except Exception as e:
            raise ValueError(f"Error loading Cipla file {file_path}: {str(e)}")

    @staticmethod
    def load_multiple_files(file_list: list) -> pd.DataFrame:
        """Load and concatenate multiple Excel files"""
        dfs = []
        for file_path in file_list:
            try:
                df = pd.read_excel(file_path)
                df['source_file'] = file_path
                dfs.append(df)
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {str(e)}")

        if not dfs:
            raise ValueError("No files could be loaded")

        return pd.concat(dfs, ignore_index=True)