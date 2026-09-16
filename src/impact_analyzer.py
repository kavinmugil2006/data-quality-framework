"""
Impact Analysis
Measure how data quality fixes affect ML model performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class ImpactAnalyzer:
    """Analyze impact of quality fixes on ML performance"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = []
        self.label_encoders = {}
    
    def _prepare_data(self, df: pd.DataFrame, target_col: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for ML (encode categoricals, split features/target)"""
        df = df.copy()
        
        if target_col is None:
            target_col = df.columns[-1]
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].copy()
        
        # Convert target to numeric
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = pd.Series(le_target.fit_transform(y.astype(str)), index=y.index)
        
        # Fill missing values first
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            X[col] = X[col].fillna(X[col].mean())
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = X[col].fillna('MISSING')
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        return X, y
    
    def evaluate_model(self, X: pd.DataFrame, y: pd.Series, model_name: str = 'rf') -> Dict:
        """Train and evaluate a single model using cross-validation"""
        
        if model_name == 'rf':
            model = RandomForestClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1, max_depth=10)
        elif model_name == 'lr':
            model = LogisticRegression(max_iter=1000, random_state=self.random_state, n_jobs=-1, solver='lbfgs', C=1.0)
        elif model_name == 'xgb':
            model = XGBClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1, use_label_encoder=False, eval_metric='logloss', verbosity=0, max_depth=5)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Ensure we have valid data
        X = X.fillna(X.mean(numeric_only=True))
        y = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 0)
        
        # Adaptive CV folds based on data size
        n_samples = len(X)
        cv_folds = min(5, max(2, n_samples // 100))  # At least 100 samples per fold
        
        # 5-fold cross-validation (or less for small datasets)
        try:
            accuracy_cv = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy', n_jobs=-1, error_score=0.0).mean()
        except Exception as e:
            print(f"      Warning: Accuracy scoring failed for {model_name}")
            accuracy_cv = 0.0
        
        try:
            f1_cv = cross_val_score(model, X, y, cv=cv_folds, scoring='f1_weighted', n_jobs=-1, error_score=0.0).mean()
        except Exception as e:
            print(f"      Warning: F1 scoring failed for {model_name}")
            f1_cv = 0.0
        
        try:
            if len(np.unique(y)) == 2:  # Binary classification
                roc_auc_cv = cross_val_score(model, X, y, cv=cv_folds, scoring='roc_auc', n_jobs=-1, error_score=0.0).mean()
            else:
                roc_auc_cv = cross_val_score(model, X, y, cv=cv_folds, scoring='roc_auc_ovr_weighted', n_jobs=-1, error_score=0.0).mean()
        except Exception as e:
            print(f"      Warning: ROC-AUC scoring failed for {model_name}")
            roc_auc_cv = 0.0
        
        return {
            'accuracy': max(accuracy_cv, 0.0),
            'f1': max(f1_cv, 0.0),
            'roc_auc': max(roc_auc_cv, 0.0),
        }
    
    def analyze_impact(self, df_original: pd.DataFrame, 
                      df_fixed: pd.DataFrame,
                      target_col: str = None,
                      models: list = ['rf', 'lr']) -> Dict:
        """
        Compare performance: original vs fixed data
        
        Returns impact metrics for each model
        """
        
        results = {}
        
        # Prepare original data
        try:
            self.label_encoders = {}
            X_orig, y_orig = self._prepare_data(df_original, target_col)
        except Exception as e:
            print(f"Error preparing original data: {e}")
            return results
        
        # Prepare fixed data with fresh encoders
        try:
            self.label_encoders = {}
            X_fixed, y_fixed = self._prepare_data(df_fixed, target_col)
        except Exception as e:
            print(f"Error preparing fixed data: {e}")
            return results

        # Ensure both datasets use the same feature columns and target encoding.
        feature_columns = sorted(set(X_orig.columns) | set(X_fixed.columns))
        X_orig = X_orig.reindex(columns=feature_columns, fill_value=0)
        X_fixed = X_fixed.reindex(columns=feature_columns, fill_value=0)

        if len(y_orig) > 0 and len(y_fixed) > 0:
            original_values = y_orig.astype(str)
            fixed_values = y_fixed.astype(str)
            label_map = {label: idx for idx, label in enumerate(sorted(set(original_values) | set(fixed_values)))}
            y_orig = original_values.map(label_map).astype(int)
            y_fixed = fixed_values.map(label_map).astype(int)
        
        # Evaluate each model
        for model_name in models:
            print(f"\n  Evaluating {model_name.upper()}...")
            
            try:
                metrics_original = self.evaluate_model(X_orig, y_orig, model_name)
                metrics_fixed = self.evaluate_model(X_fixed, y_fixed, model_name)
                
                # Calculate improvement
                improvement = {
                    'accuracy': metrics_fixed['accuracy'] - metrics_original['accuracy'],
                    'f1': metrics_fixed['f1'] - metrics_original['f1'],
                    'roc_auc': metrics_fixed['roc_auc'] - metrics_original['roc_auc'],
                }
                
                results[model_name] = {
                    'original': metrics_original,
                    'fixed': metrics_fixed,
                    'improvement': improvement,
                }
            except Exception as e:
                print(f"    Error evaluating {model_name}: {e}")
                continue
        
        return results
    
    def analyze_quality_issue_impact(self, df: pd.DataFrame, 
                                     issue_type: str,
                                     target_col: str = None,
                                     models: list = ['rf']) -> Dict:
        """
        Analyze impact of a SINGLE quality issue on ML performance
        
        Simulates the issue in clean data and measures F1 drop
        """
        
        df_clean = df.copy()
        df_with_issue = df.copy()
        
        if issue_type == 'missing_values':
            # Introduce missing values (10% of data)
            for col in df_with_issue.select_dtypes(include=['number']).columns:
                missing_idx = np.random.choice(len(df_with_issue), int(0.1 * len(df_with_issue)), replace=False)
                df_with_issue.iloc[missing_idx, df_with_issue.columns.get_loc(col)] = np.nan
            
            # Remove missing
            df_with_issue = df_with_issue.dropna()
        
        elif issue_type == 'outliers':
            # Introduce outliers
            numeric_cols = df_with_issue.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                outlier_idx = np.random.choice(len(df_with_issue), int(0.05 * len(df_with_issue)), replace=False)
                df_with_issue.iloc[outlier_idx, df_with_issue.columns.get_loc(col)] *= 5
        
        elif issue_type == 'duplicates':
            # Introduce duplicates (5%)
            n_dups = int(0.05 * len(df_with_issue))
            dup_rows = df_with_issue.sample(n=n_dups, random_state=42)
            df_with_issue = pd.concat([df_with_issue, dup_rows], ignore_index=True)
        
        return self.analyze_impact(df_with_issue, df_clean, target_col, models)
    
    def print_impact_report(self, results: Dict, df_original, df_fixed):
        """Print formatted impact report"""
        print("\n" + "="*70)
        print("IMPACT ANALYSIS REPORT")
        print("="*70)
        print(f"Original data shape: {df_original.shape}")
        print(f"Fixed data shape: {df_fixed.shape}")
        print(f"Rows removed/changed: {len(df_original) - len(df_fixed)}")
        print("="*70)
        
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()} Model:")
            print(f"  Original F1:   {metrics['original']['f1']:.4f}")
            print(f"  Fixed F1:      {metrics['fixed']['f1']:.4f}")
            
            # Safe division - avoid dividing by zero
            if metrics['original']['f1'] > 0:
                f1_pct = (metrics['improvement']['f1'] / metrics['original']['f1']) * 100
                print(f"  F1 Improvement: {metrics['improvement']['f1']:+.4f} ({f1_pct:+.2f}%)")
            else:
                print(f"  F1 Improvement: {metrics['improvement']['f1']:+.4f} (undefined %)")
            
            print(f"\n  Original Accuracy: {metrics['original']['accuracy']:.4f}")
            print(f"  Fixed Accuracy:    {metrics['fixed']['accuracy']:.4f}")
            print(f"  Accuracy Improvement: {metrics['improvement']['accuracy']:+.4f}")
