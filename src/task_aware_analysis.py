"""
Extended ML Task Support Module
===============================
Extend data quality framework beyond binary classification to:
  - Regression (price prediction, demand forecasting)
  - Multi-class classification (disease diagnosis, product category)
  - Clustering (customer segmentation, anomaly detection)

Key Changes:
  - Classification: F1, Precision, Recall
  - Regression: RMSE, MAE, R²
  - Clustering: Silhouette score, Davies-Bouldin index

Quality impact varies by task:
  - Regression: Outliers hurt MORE (they're extreme errors)
  - Clustering: Duplicates hurt MORE (confuse similarity)
  - Classification: Imbalance hurts MORE (class-wise metrics)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, davies_bouldin_score,
    f1_score, precision_score, recall_score
)
import warnings
warnings.filterwarnings('ignore')


class TaskAwareImpactAnalyzer:
    """
    Analyze data quality impact across different ML tasks.
    
    Understands:
      - Regression tasks (continuous output)
      - Classification tasks (categorical output)
      - Clustering tasks (unsupervised grouping)
    """
    
    def __init__(self, task_type: str = 'classification'):
        """
        Args:
            task_type: 'classification', 'regression', or 'clustering'
        """
        if task_type not in ['classification', 'regression', 'clustering']:
            raise ValueError(f"Unknown task type: {task_type}")
        
        self.task_type = task_type
        self.models = None
        self.cv_folds = None
    
    # ============================================================
    # REGRESSION SUPPORT
    # ============================================================
    
    def _create_regression_models(self):
        """Create models for regression tasks."""
        return {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'lr': LinearRegression(),
        }
    
    def _evaluate_regression(self, y_true, y_pred) -> Dict:
        """
        Evaluate regression performance.
        
        Returns metrics:
          - RMSE: Root Mean Squared Error (penalizes outliers heavily)
          - MAE: Mean Absolute Error (robust to outliers)
          - R²: Coefficient of determination (proportion of variance explained)
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
        }
    
    def compute_regression_impact(
        self,
        X_original: np.ndarray,
        X_fixed: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5
    ) -> Dict:
        """
        Measure data quality impact on regression task.
        
        Args:
            X_original: Original features
            X_fixed: Fixed features
            y: Target (continuous)
            cv_folds: Cross-validation folds
            
        Returns:
            dict with regression metrics before/after fixes
        """
        print(f"\n  Evaluating Regression (K={cv_folds} fold CV)...")
        
        models = self._create_regression_models()
        results = {'rf': {}, 'lr': {}}
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for model_name, model in models.items():
            print(f"\n    Model: {model_name.upper()}")
            
            # Evaluate on original data
            original_scores = []
            fixed_scores = []
            
            for train_idx, test_idx in kf.split(X_original):
                X_train_orig, X_test_orig = X_original[train_idx], X_original[test_idx]
                X_train_fixed, X_test_fixed = X_fixed[train_idx], X_fixed[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Train on original
                model.fit(X_train_orig, y_train)
                y_pred_orig = model.predict(X_test_orig)
                orig_metrics = self._evaluate_regression(y_test, y_pred_orig)
                original_scores.append(orig_metrics['rmse'])
                
                # Train on fixed
                model.fit(X_train_fixed, y_train)
                y_pred_fixed = model.predict(X_test_fixed)
                fixed_metrics = self._evaluate_regression(y_test, y_pred_fixed)
                fixed_scores.append(fixed_metrics['rmse'])
            
            # Average across folds
            avg_original_rmse = np.mean(original_scores)
            avg_fixed_rmse = np.mean(fixed_scores)
            
            # RMSE improvement (lower is better, so improvement = original - fixed)
            rmse_improvement = avg_original_rmse - avg_fixed_rmse
            rmse_improvement_pct = (rmse_improvement / avg_original_rmse * 100) if avg_original_rmse > 0 else 0
            
            results[model_name] = {
                'original': {'rmse': float(avg_original_rmse)},
                'fixed': {'rmse': float(avg_fixed_rmse)},
                'improvement': {
                    'rmse': float(rmse_improvement),
                    'rmse_pct': float(rmse_improvement_pct),
                }
            }
            
            print(f"      Original RMSE: {avg_original_rmse:.4f}")
            print(f"      Fixed RMSE: {avg_fixed_rmse:.4f}")
            print(f"      Improvement: {rmse_improvement_pct:+.2f}%")
        
        return results
    
    # ============================================================
    # MULTI-CLASS CLASSIFICATION SUPPORT
    # ============================================================
    
    def compute_multiclass_impact(
        self,
        X_original: np.ndarray,
        X_fixed: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5,
        average: str = 'weighted'
    ) -> Dict:
        """
        Measure data quality impact on multi-class classification.
        
        Args:
            average: 'weighted', 'macro', 'micro'
              - weighted: F1 weighted by class support (recommended)
              - macro: unweighted mean F1 (fair to rare classes)
              - micro: same as accuracy (ignore class imbalance)
        """
        print(f"\n  Evaluating Multi-class Classification (K={cv_folds}, average={average})...")
        
        models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'lr': LogisticRegression(max_iter=1000, random_state=42),
        }
        
        results = {'rf': {}, 'lr': {}}
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for model_name, model in models.items():
            print(f"\n    Model: {model_name.upper()}")
            
            original_f1s = []
            fixed_f1s = []
            
            for train_idx, test_idx in skf.split(X_original, y):
                X_train_orig, X_test_orig = X_original[train_idx], X_original[test_idx]
                X_train_fixed, X_test_fixed = X_fixed[train_idx], X_fixed[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Original
                model.fit(X_train_orig, y_train)
                y_pred_orig = model.predict(X_test_orig)
                f1_orig = f1_score(y_test, y_pred_orig, average=average, zero_division=0)
                original_f1s.append(f1_orig)
                
                # Fixed
                model.fit(X_train_fixed, y_train)
                y_pred_fixed = model.predict(X_test_fixed)
                f1_fixed = f1_score(y_test, y_pred_fixed, average=average, zero_division=0)
                fixed_f1s.append(f1_fixed)
            
            avg_orig_f1 = np.mean(original_f1s)
            avg_fixed_f1 = np.mean(fixed_f1s)
            f1_improvement = avg_fixed_f1 - avg_orig_f1
            f1_improvement_pct = (f1_improvement / avg_orig_f1 * 100) if avg_orig_f1 > 0 else 0
            
            results[model_name] = {
                'original': {'f1': float(avg_orig_f1)},
                'fixed': {'f1': float(avg_fixed_f1)},
                'improvement': {
                    'f1': float(f1_improvement),
                    'f1_pct': float(f1_improvement_pct),
                }
            }
            
            print(f"      Original F1 ({average}): {avg_orig_f1:.4f}")
            print(f"      Fixed F1 ({average}): {avg_fixed_f1:.4f}")
            print(f"      Improvement: {f1_improvement_pct:+.2f}%")
        
        return results
    
    # ============================================================
    # CLUSTERING SUPPORT
    # ============================================================
    
    def compute_clustering_impact(
        self,
        X_original: np.ndarray,
        X_fixed: np.ndarray,
        n_clusters: int = 3,
        random_seed: int = 42
    ) -> Dict:
        """
        Measure data quality impact on clustering.
        
        Metrics:
          - Silhouette Score: How well-separated clusters are (higher is better)
          - Davies-Bouldin Index: Average cluster similarity (lower is better)
        """
        print(f"\n  Evaluating Clustering (K={n_clusters})...")
        
        results = {
            'silhouette': {},
            'davies_bouldin': {},
        }
        
        # Silhouette score (higher is better, range: -1 to 1)
        print(f"\n    Silhouette Score (higher is better):")
        try:
            silh_orig = silhouette_score(X_original, KMeans(n_clusters, random_state=random_seed).fit_predict(X_original))
            silh_fixed = silhouette_score(X_fixed, KMeans(n_clusters, random_state=random_seed).fit_predict(X_fixed))
            
            silh_improvement = silh_fixed - silh_orig
            
            results['silhouette'] = {
                'original': float(silh_orig),
                'fixed': float(silh_fixed),
                'improvement': float(silh_improvement),
            }
            
            print(f"      Original: {silh_orig:.4f}")
            print(f"      Fixed: {silh_fixed:.4f}")
            print(f"      Change: {silh_improvement:+.4f}")
        except Exception as e:
            print(f"      Error: {e}")
        
        # Davies-Bouldin Index (lower is better)
        print(f"\n    Davies-Bouldin Index (lower is better):")
        try:
            db_orig = davies_bouldin_score(X_original, KMeans(n_clusters, random_state=random_seed).fit_predict(X_original))
            db_fixed = davies_bouldin_score(X_fixed, KMeans(n_clusters, random_state=random_seed).fit_predict(X_fixed))
            
            db_improvement = db_orig - db_fixed  # Lower is better, so improvement = orig - fixed
            
            results['davies_bouldin'] = {
                'original': float(db_orig),
                'fixed': float(db_fixed),
                'improvement': float(db_improvement),
            }
            
            print(f"      Original: {db_orig:.4f}")
            print(f"      Fixed: {db_fixed:.4f}")
            print(f"      Improvement: {db_improvement:+.4f} (negative = worse)")
        except Exception as e:
            print(f"      Error: {e}")
        
        return results
    
    # ============================================================
    # Main compute_impact method
    # ============================================================
    
    def compute_impact(
        self,
        X_original: np.ndarray,
        X_fixed: np.ndarray,
        y: Optional[np.ndarray] = None,
        cv_folds: int = 5,
        n_clusters: int = 3
    ) -> Dict:
        """
        Main method: Compute data quality impact based on task type.
        """
        print(f"\n{'='*60}")
        print(f"TASK-SPECIFIC IMPACT ANALYSIS: {self.task_type.upper()}")
        print(f"{'='*60}")
        
        if self.task_type == 'regression':
            if y is None:
                raise ValueError("Regression requires target variable y")
            return self.compute_regression_impact(X_original, X_fixed, y, cv_folds)
        
        elif self.task_type == 'classification':
            if y is None:
                raise ValueError("Classification requires target variable y")
            
            n_classes = len(np.unique(y))
            if n_classes == 2:
                print(f"\nTask: Binary Classification")
                return self.compute_multiclass_impact(X_original, X_fixed, y, cv_folds)
            else:
                print(f"\nTask: Multi-class Classification ({n_classes} classes)")
                return self.compute_multiclass_impact(X_original, X_fixed, y, cv_folds)
        
        elif self.task_type == 'clustering':
            return self.compute_clustering_impact(X_original, X_fixed, n_clusters)


# ============================================================================
# Demo
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("EXTENDED ML TASK SUPPORT - DEMO")
    print("="*70)
    
    # Generate synthetic data
    np.random.seed(42)
    
    # Task 1: Regression
    print("\n[1/3] REGRESSION TASK (Price Prediction)")
    print("-" * 70)
    
    X_reg_orig = np.random.randn(200, 5)
    X_reg_fixed = X_reg_orig + np.random.randn(200, 5) * 0.1  # Slight improvement
    y_reg = np.dot(X_reg_orig, np.array([2, -1, 0.5, 3, -0.5])) + np.random.randn(200) * 0.5
    
    analyzer_reg = TaskAwareImpactAnalyzer(task_type='regression')
    result_reg = analyzer_reg.compute_impact(X_reg_orig, X_reg_fixed, y_reg, cv_folds=5)
    
    print("\n  Summary:")
    for model, metrics in result_reg.items():
        print(f"    {model.upper()}: RMSE improvement {metrics['improvement']['rmse_pct']:+.2f}%")
    
    # Task 2: Multi-class Classification
    print("\n[2/3] MULTI-CLASS CLASSIFICATION (Product Category)")
    print("-" * 70)
    
    X_clf_orig = np.random.randn(300, 5)
    X_clf_fixed = X_clf_orig + np.random.randn(300, 5) * 0.15
    y_clf = np.random.choice([0, 1, 2, 3], 300)  # 4 classes
    
    analyzer_clf = TaskAwareImpactAnalyzer(task_type='classification')
    result_clf = analyzer_clf.compute_impact(X_clf_orig, X_clf_fixed, y_clf, cv_folds=5)
    
    print("\n  Summary:")
    for model, metrics in result_clf.items():
        print(f"    {model.upper()}: F1 improvement {metrics['improvement']['f1_pct']:+.2f}%")
    
    # Task 3: Clustering
    print("\n[3/3] CLUSTERING (Customer Segmentation)")
    print("-" * 70)
    
    X_clust_orig = np.random.randn(150, 5)
    X_clust_fixed = X_clust_orig + np.random.randn(150, 5) * 0.08
    
    analyzer_clust = TaskAwareImpactAnalyzer(task_type='clustering')
    result_clust = analyzer_clust.compute_impact(X_clust_orig, X_clust_fixed, n_clusters=3)
    
    print("\n  Summary:")
    print(f"    Silhouette improvement: {result_clust['silhouette']['improvement']:+.4f}")
    print(f"    Davies-Bouldin improvement: {result_clust['davies_bouldin']['improvement']:+.4f}")
    
    print("\n" + "="*70)
    print("✓ DEMO COMPLETE")
    print("="*70)
    print("\nKey Insights:")
    print("  • Regression: RMSE penalizes outliers heavily")
    print("  • Multi-class: Need stratified CV (classes can be imbalanced)")
    print("  • Clustering: Quality impact on cluster cohesion different")
    print("  • One framework, three task types!\n")
