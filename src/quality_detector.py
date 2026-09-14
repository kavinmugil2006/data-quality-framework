"""
Data Quality Detection Framework
Automated detection of data quality issues and their impact on ML performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class QualityDetector:
    """Detects data quality issues in tabular datasets"""
    
    def __init__(self):
        self.issues = {}
        self.quality_scores = {}
        
    def detect_all(self, df: pd.DataFrame) -> Dict:
        """Run all quality checks"""
        
        results = {
            'completeness': self._check_completeness(df),
            'consistency': self._check_consistency(df),
            'validity': self._check_validity(df),
            'uniqueness': self._check_uniqueness(df),
            'outliers': self._check_outliers(df),
            'imbalance': self._check_imbalance(df),
            'correlation': self._check_correlation(df),
        }
        
        return results
    
    def _check_completeness(self, df: pd.DataFrame) -> Dict:
        """Check for missing values"""
        missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()
        
        return {
            'name': 'Missing Values',
            'severity': 'critical' if any(v > 30 for v in missing_pct.values()) else 'medium',
            'details': {k: v for k, v in missing_pct.items() if v > 0},
            'overall_missing_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100),
        }
    
    def _check_consistency(self, df: pd.DataFrame) -> Dict:
        """Check for duplicates and type inconsistencies"""
        n_duplicates = df.duplicated().sum()
        dup_pct = (n_duplicates / len(df) * 100)
        
        return {
            'name': 'Duplicates',
            'severity': 'high' if dup_pct > 5 else 'low',
            'count': n_duplicates,
            'percentage': dup_pct,
        }
    
    def _check_validity(self, df: pd.DataFrame) -> Dict:
        """Check for invalid values in categorical columns"""
        issues = {}
        
        for col in df.select_dtypes(include='object').columns:
            n_unique = df[col].nunique()
            n_rows = len(df)
            if n_unique == n_rows:  # Each value is unique
                issues[col] = f"Potentially ID column or high cardinality"
        
        return {
            'name': 'Validity Issues',
            'severity': 'medium' if issues else 'low',
            'details': issues,
        }
    
    def _check_uniqueness(self, df: pd.DataFrame) -> Dict:
        """Check for duplicate rows"""
        return {
            'name': 'Row Uniqueness',
            'duplicate_rows': df.duplicated().sum(),
            'unique_rows_pct': ((len(df) - df.duplicated().sum()) / len(df) * 100),
        }
    
    def _check_outliers(self, df: pd.DataFrame) -> Dict:
        """Detect outliers in numeric columns using IQR"""
        outlier_summary = {}
        
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            n_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_pct = (n_outliers / len(df) * 100)
            
            if n_outliers > 0:
                outlier_summary[col] = {
                    'count': n_outliers,
                    'percentage': outlier_pct,
                }
        
        return {
            'name': 'Outliers (IQR)',
            'severity': 'high' if any(v['percentage'] > 10 for v in outlier_summary.values()) else 'medium',
            'details': outlier_summary,
            'total_outlier_records': sum(v['count'] for v in outlier_summary.values()),
        }
    
    def _check_imbalance(self, df: pd.DataFrame, target_col: str = None) -> Dict:
        """Check class imbalance (assumes last column is target)"""
        
        # Try to find target column
        if target_col is None:
            target_col = df.columns[-1]
        
        if target_col in df.columns:
            if df[target_col].dtype in ['object', 'category'] or df[target_col].nunique() < 20:
                value_counts = df[target_col].value_counts(normalize=True)
                imbalance_ratio = value_counts.max() / value_counts.min()
                
                return {
                    'name': 'Class Imbalance',
                    'severity': 'high' if imbalance_ratio > 5 else 'medium' if imbalance_ratio > 2 else 'low',
                    'imbalance_ratio': imbalance_ratio,
                    'distribution': value_counts.to_dict(),
                }
        
        return {
            'name': 'Class Imbalance',
            'severity': 'low',
            'imbalance_ratio': 1.0,
        }
    
    def _check_correlation(self, df: pd.DataFrame) -> Dict:
        """Check for high correlation between features"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return {'name': 'Feature Correlation', 'high_corr_pairs': 0}
        
        corr_matrix = numeric_df.corr().abs()
        
        # Find highly correlated pairs (>0.9)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.9:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': float(corr_matrix.iloc[i, j]),
                    })
        
        return {
            'name': 'Feature Correlation',
            'high_corr_pairs': len(high_corr_pairs),
            'details': high_corr_pairs,
        }
    
    def generate_quality_score(self, results: Dict) -> float:
        """Generate 0-100 quality score"""
        
        score = 100.0
        
        # Penalize for missing values
        missing_pct = results['completeness']['overall_missing_pct']
        score -= min(missing_pct * 0.5, 25)  # Max -25 points
        
        # Penalize for duplicates
        dup_pct = results['consistency']['percentage']
        score -= min(dup_pct * 2, 15)  # Max -15 points
        
        # Penalize for outliers
        total_outliers = results['outliers']['total_outlier_records']
        if total_outliers > 0:
            score -= 10
        
        # Penalize for imbalance
        imbalance = results['imbalance'].get('imbalance_ratio', 1.0)
        if imbalance > 5:
            score -= 15
        elif imbalance > 2:
            score -= 10
        
        # Penalize for high correlation
        if results['correlation']['high_corr_pairs'] > 3:
            score -= 10
        
        return max(score, 0)
    
    def print_report(self, df: pd.DataFrame, results: Dict):
        """Print quality report"""
        quality_score = self.generate_quality_score(results)
        
        print("\n" + "="*60)
        print(f"DATA QUALITY REPORT")
        print("="*60)
        print(f"Dataset shape: {df.shape}")
        print(f"Overall Quality Score: {quality_score:.1f}/100")
        print("="*60)
        
        for check_name, check_result in results.items():
            print(f"\n{check_result['name']}:")
            print(f"  Severity: {check_result.get('severity', 'N/A')}")
            for key, value in check_result.items():
                if key not in ['name', 'severity', 'details']:
                    if isinstance(value, float):
                        print(f"  {key}: {value:.2f}")
                    else:
                        print(f"  {key}: {value}")