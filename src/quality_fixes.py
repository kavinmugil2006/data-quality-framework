"""
Data Quality Fixes
Strategies to fix detected quality issues
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple

class QualityFixer:
    """Apply fixes to data quality issues"""
    
    def __init__(self):
        self.fixes_applied = {}
    
    def fix_missing_values(self, df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
        """
        Fix missing values
        
        Strategies:
        - 'drop': Remove rows with missing values
        - 'mean': Fill with mean (numeric)
        - 'median': Fill with median (numeric)
        - 'forward_fill': Forward fill
        """
        df = df.copy()
        
        if strategy == 'drop':
            df = df.dropna().reset_index(drop=True)
        elif strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == 'forward_fill':
            df = df.ffill().bfill()
        
        self.fixes_applied['missing_values'] = strategy
        return df
    
    def remove_duplicates(self, df: pd.DataFrame, keep: str = 'first') -> pd.DataFrame:
        """Remove duplicate rows"""
        df = df.copy()
        initial_rows = len(df)
        df = df.drop_duplicates(keep=keep)
        final_rows = len(df)
        
        self.fixes_applied['duplicates'] = f"Removed {initial_rows - final_rows} rows"
        return df
    
    def remove_outliers(self, df: pd.DataFrame, method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers
        
        Methods:
        - 'iqr': Interquartile range (default)
        - 'zscore': Z-score > 3
        """
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if method == 'iqr':
            mask = pd.Series([True] * len(df), index=df.index)
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                col_mask = (df[col] >= lower) & (df[col] <= upper)
                mask = mask & col_mask
            
            initial_rows = len(df)
            df = df.loc[mask]
            self.fixes_applied['outliers'] = f"Removed {initial_rows - len(df)} rows"
        
        elif method == 'zscore':
            from scipy import stats
            mask = pd.Series([True] * len(df), index=df.index)
            for col in numeric_cols:
                col_filled = df[col].fillna(df[col].mean())
                col_mask = np.abs(stats.zscore(col_filled)) < 3
                mask = mask & col_mask
            
            initial_rows = len(df)
            df = df.loc[mask]
            self.fixes_applied['outliers'] = f"Removed {initial_rows - len(df)} rows"
        
        return df.reset_index(drop=True)
    
    def fix_imbalance(self, df: pd.DataFrame, target_col: str = None, method: str = 'undersample') -> pd.DataFrame:
        """
        Fix class imbalance
        
        Methods:
        - 'undersample': Reduce majority class
        - 'oversample': Duplicate minority class (basic)
        """
        df = df.copy()
        
        if target_col is None:
            target_col = df.columns[-1]
        
        if target_col not in df.columns:
            return df
        
        # Check if imbalance is severe enough to fix
        class_counts = df[target_col].value_counts()
        imbalance_ratio = class_counts.max() / class_counts.min()
        
        # Only fix if imbalance is > 10:1
        if imbalance_ratio <= 10:
            return df
        
        if method == 'undersample':
            # Don't undersample too aggressively - use 2x minority count
            min_count = class_counts.min()
            target_count = min(int(min_count * 3), class_counts.max())  # Compromise between both
            
            df_balanced = pd.concat([
                df[df[target_col] == cls].sample(n=min(target_count, len(df[df[target_col] == cls])), random_state=42)
                for cls in df[target_col].unique()
            ])
            
            initial_rows = len(df)
            final_rows = len(df_balanced)
            self.fixes_applied['imbalance'] = f"Rebalanced from {initial_rows} to {final_rows}"
            return df_balanced.reset_index(drop=True)
        
        elif method == 'oversample':
            # Only oversample minority to 50% of majority
            class_counts = df[target_col].value_counts()
            max_count = class_counts.max()
            
            dfs = []
            for cls in df[target_col].unique():
                cls_df = df[df[target_col] == cls]
                if len(cls_df) < max_count * 0.5:
                    cls_df = cls_df.sample(n=int(max_count * 0.5), replace=True, random_state=42)
                dfs.append(cls_df)
            
            df_balanced = pd.concat(dfs, ignore_index=True)
            self.fixes_applied['imbalance'] = f"Rebalanced to {len(df_balanced)}"
            return df_balanced.reset_index(drop=True)
        
        return df
    
    def scale_features(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """Standardize numeric features while leaving the target column untouched."""
        df = df.copy()

        if target_col is None:
            target_col = df.columns[-1] if len(df.columns) > 0 else None

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col is not None and target_col in numeric_cols:
            numeric_cols = [col for col in numeric_cols if col != target_col]

        if not numeric_cols:
            self.fixes_applied['scaling'] = 'No numeric features to scale'
            return df

        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        self.fixes_applied['scaling'] = 'StandardScaler applied'
        return df
    
    def apply_fixes(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """
        Apply multiple fixes in sequence
        
        config = {
            'missing_values': 'drop',
            'duplicates': True,
            'outliers': True,
            'imbalance': True,
            'scale': True,
        }
        """
        df = df.copy()
        
        if config.get('missing_values'):
            df = self.fix_missing_values(df, strategy=config['missing_values'])
        
        if config.get('duplicates'):
            df = self.remove_duplicates(df)
        
        if config.get('outliers'):
            df = self.remove_outliers(df, method='iqr')
        
        if config.get('imbalance'):
            df = self.fix_imbalance(df, method='undersample')
        
        if config.get('scale'):
            target_col = df.columns[-1] if len(df.columns) > 0 else None
            df = self.scale_features(df, target_col=target_col)
        
        return df
    
    def get_fixes_summary(self) -> Dict:
        """Get summary of applied fixes"""
        return self.fixes_applied
