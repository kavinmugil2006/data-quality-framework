"""
Predictive Impact Modeling Module
==================================
Meta-model to predict which data quality issues will hurt ML performance.
Instead of trying all fixes, predict impact first!

Key Idea:
  Quality metrics (missing%, duplicates%, outliers%, etc) → Predicted F1 Impact
  
This saves compute: predict impact in seconds vs waiting for 5-fold CV!
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import pickle
import json
from pathlib import Path

class PredictiveImpactModel:
    """
    Meta-model that learns relationship between quality issues and ML performance.
    
    Training data: (quality_metrics) → (F1_improvement)
    Usage: Given new dataset's quality metrics, predict expected F1 improvement
    """
    
    def __init__(self, model_type='rf'):
        """
        Args:
            model_type: 'rf' (Random Forest), 'gb' (Gradient Boosting), or 'lr' (Linear)
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.is_trained = False
        
        # Initialize model
        if model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'gb':
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_type == 'lr':
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def extract_quality_features(self, quality_report):
        """
        Convert quality detection report to feature vector.
        
        Args:
            quality_report: Dict from QualityDetector.analyze()
            
        Returns:
            np.array of shape (1, n_features)
            
        Features (11 dimensions):
          1. Missing values (%)
          2. Duplicates (%)
          3. Outliers (%)
          4. Class imbalance ratio
          5. High correlation features (%)
          6. Type inconsistencies (count)
          7. Data quality score (0-100)
          8. Dataset size (log scale)
          9. Number of features
          10. Categorical features (%)
          11. Numeric features (%)
        """
        features = [
            quality_report.get('missing_values', 0),
            quality_report.get('duplicates', 0),
            quality_report.get('outliers', 0),
            quality_report.get('imbalance_ratio', 1.0),
            quality_report.get('high_correlation_pct', 0),
            quality_report.get('type_inconsistencies', 0),
            quality_report.get('quality_score', 50),
            np.log1p(quality_report.get('dataset_size', 1000)),
            quality_report.get('n_features', 10),
            quality_report.get('categorical_pct', 0),
            quality_report.get('numeric_pct', 100),
        ]
        
        self.feature_names = [
            'missing_values', 'duplicates', 'outliers', 'imbalance_ratio',
            'high_correlation_pct', 'type_inconsistencies', 'quality_score',
            'log_dataset_size', 'n_features', 'categorical_pct', 'numeric_pct'
        ]
        
        return np.array(features).reshape(1, -1)
    
    def train(self, training_data):
        """
        Train meta-model on historical data.
        
        Args:
            training_data: List of dicts with keys:
              - 'quality_report': Quality metrics from QualityDetector
              - 'f1_improvement': Actual F1 improvement from ImpactAnalyzer
              
        Example:
            training_data = [
                {
                    'quality_report': {'missing_values': 2.5, 'duplicates': 0.1, ...},
                    'f1_improvement': 0.015  # 1.5% improvement
                },
                ...
            ]
        """
        if not training_data:
            raise ValueError("Training data cannot be empty")
        
        # Extract features and targets
        X = []
        y = []
        
        for example in training_data:
            quality_report = example['quality_report']
            f1_improvement = example['f1_improvement']
            
            # Extract features for this example
            features = self.extract_quality_features(quality_report).flatten()
            X.append(features)
            y.append(f1_improvement)
        
        X = np.array(X)
        y = np.array(y)
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Report training metrics
        train_r2 = self.model.score(X_scaled, y)
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='r2')
        
        print(f"\n{'='*60}")
        print(f"PREDICTIVE IMPACT MODEL TRAINED")
        print(f"{'='*60}")
        print(f"Model Type: {self.model_type.upper()}")
        print(f"Training Samples: {len(X)}")
        print(f"Training R² Score: {train_r2:.4f}")
        print(f"Cross-Val R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"Features: {len(self.feature_names)}")
        print(f"{'='*60}\n")
        
        return {
            'train_r2': float(train_r2),
            'cv_r2_mean': float(cv_scores.mean()),
            'cv_r2_std': float(cv_scores.std()),
            'n_samples': len(X),
            'n_features': len(self.feature_names)
        }
    
    def predict(self, quality_report):
        """
        Predict F1 improvement for new dataset based on its quality metrics.
        
        Args:
            quality_report: Dict from QualityDetector.analyze()
            
        Returns:
            dict with:
              - 'predicted_f1_improvement': Predicted change in F1
              - 'confidence_interval': (lower, upper) bounds
              - 'recommendation': 'Worth fixing' or 'Minimal benefit'
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction!")
        
        # Extract features
        X = self.extract_quality_features(quality_report)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predicted_improvement = self.model.predict(X_scaled)[0]
        
        # Get feature importance (if available)
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            feature_importance = {k: float(v) for k, v in sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]}  # Top 5 most important features
        
        # Confidence interval (rough estimate: ±0.02 for RF/GB, ±0.05 for LR)
        if self.model_type in ['rf', 'gb']:
            margin = 0.02
        else:
            margin = 0.05
        
        lower_bound = predicted_improvement - margin
        upper_bound = predicted_improvement + margin
        
        # Recommendation
        if abs(predicted_improvement) > 0.01:  # >1% improvement
            recommendation = "✓ Worth fixing"
            severity = "HIGH"
        elif abs(predicted_improvement) > 0.005:  # >0.5% improvement
            recommendation = "~ Maybe fix (marginal benefit)"
            severity = "MEDIUM"
        else:
            recommendation = "✗ Skip (minimal benefit)"
            severity = "LOW"
        
        return {
            'predicted_f1_improvement': float(predicted_improvement),
            'predicted_improvement_pct': float(predicted_improvement * 100),
            'confidence_interval': (float(lower_bound), float(upper_bound)),
            'recommendation': recommendation,
            'severity': severity,
            'top_factors': feature_importance,
            'quality_score': quality_report.get('quality_score', 0)
        }
    
    def save(self, filepath):
        """Save trained model to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model!")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist(),
        }
        
        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump((self.model, model_data), f)
        
        print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath):
        """Load trained model from disk."""
        filepath = Path(filepath)
        
        with open(filepath, 'rb') as f:
            self.model, model_data = pickle.load(f)
        
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.scaler.mean_ = np.array(model_data['scaler_mean'])
        self.scaler.scale_ = np.array(model_data['scaler_scale'])
        self.is_trained = True
        
        print(f"✓ Model loaded from {filepath}")


# ============================================================================
# Example Usage & Training
# ============================================================================

def create_synthetic_training_data():
    """
    Create synthetic training data for demonstration.
    Real data would come from historical project impact analyses.
    """
    np.random.seed(42)
    
    training_data = []
    
    for i in range(50):
        # Generate random quality metrics
        missing = np.random.uniform(0, 10)
        duplicates = np.random.uniform(0, 5)
        outliers = np.random.uniform(0, 30)
        imbalance = np.random.uniform(1, 10)
        
        # Quality score (lower score = more issues)
        quality_score = 100 - (missing * 2 + duplicates * 3 + outliers * 0.5)
        quality_score = max(30, min(100, quality_score))
        
        # Simulate F1 improvement (simplified: more issues → larger potential improvement)
        # But diminishing returns (very dirty data is harder to fix)
        issues_severity = (missing + duplicates * 2 + outliers * 0.3) / 100
        if quality_score > 80:
            f1_improvement = np.random.uniform(-0.01, 0.02)  # Clean data: minimal benefit
        elif quality_score > 50:
            f1_improvement = np.random.uniform(0.01, 0.05)  # Moderate: good potential
        else:
            f1_improvement = np.random.uniform(0.00, 0.03)  # Dirty: harder to improve
        
        training_example = {
            'quality_report': {
                'missing_values': missing,
                'duplicates': duplicates,
                'outliers': outliers,
                'imbalance_ratio': imbalance,
                'high_correlation_pct': np.random.uniform(0, 20),
                'type_inconsistencies': np.random.randint(0, 3),
                'quality_score': quality_score,
                'dataset_size': np.random.randint(1000, 100000),
                'n_features': np.random.randint(5, 50),
                'categorical_pct': np.random.uniform(20, 80),
                'numeric_pct': None,  # Will be calculated
            },
            'f1_improvement': f1_improvement
        }
        
        training_example['quality_report']['numeric_pct'] = \
            100 - training_example['quality_report']['categorical_pct']
        
        training_data.append(training_example)
    
    return training_data


if __name__ == '__main__':
    print("\n" + "="*60)
    print("PREDICTIVE IMPACT MODELING - DEMO")
    print("="*60)
    
    # Create training data
    print("\n[1/4] Creating synthetic training data...")
    training_data = create_synthetic_training_data()
    print(f"  ✓ Generated {len(training_data)} training examples")
    
    # Train model
    print("\n[2/4] Training predictive model...")
    predictor = PredictiveImpactModel(model_type='rf')
    metrics = predictor.train(training_data)
    
    # Test prediction
    print("\n[3/4] Testing predictions on example datasets...")
    
    # Example 1: Clean dataset (high quality)
    clean_data = {
        'missing_values': 0.5,
        'duplicates': 0.1,
        'outliers': 5.0,
        'imbalance_ratio': 1.2,
        'high_correlation_pct': 5,
        'type_inconsistencies': 0,
        'quality_score': 95,
        'dataset_size': 50000,
        'n_features': 20,
        'categorical_pct': 30,
        'numeric_pct': 70,
    }
    
    print("\n  Test Case 1: CLEAN DATASET")
    print(f"    Quality Score: {clean_data['quality_score']}/100")
    result = predictor.predict(clean_data)
    print(f"    Predicted F1 Improvement: {result['predicted_improvement_pct']:.2f}%")
    print(f"    Recommendation: {result['recommendation']}")
    
    # Example 2: Messy dataset (low quality)
    messy_data = {
        'missing_values': 8.0,
        'duplicates': 3.5,
        'outliers': 25.0,
        'imbalance_ratio': 8.0,
        'high_correlation_pct': 35,
        'type_inconsistencies': 5,
        'quality_score': 45,
        'dataset_size': 5000,
        'n_features': 30,
        'categorical_pct': 60,
        'numeric_pct': 40,
    }
    
    print("\n  Test Case 2: MESSY DATASET")
    print(f"    Quality Score: {messy_data['quality_score']}/100")
    result = predictor.predict(messy_data)
    print(f"    Predicted F1 Improvement: {result['predicted_improvement_pct']:.2f}%")
    print(f"    Recommendation: {result['recommendation']}")
    print(f"    Top Factors:")
    if result['top_factors']:
        for factor, importance in result['top_factors'].items():
            print(f"      - {factor}: {importance:.4f}")
    
    # Example 3: Moderately problematic
    moderate_data = {
        'missing_values': 3.0,
        'duplicates': 1.5,
        'outliers': 12.0,
        'imbalance_ratio': 3.5,
        'high_correlation_pct': 15,
        'type_inconsistencies': 2,
        'quality_score': 72,
        'dataset_size': 15000,
        'n_features': 15,
        'categorical_pct': 40,
        'numeric_pct': 60,
    }
    
    print("\n  Test Case 3: MODERATE QUALITY")
    print(f"    Quality Score: {moderate_data['quality_score']}/100")
    result = predictor.predict(moderate_data)
    print(f"    Predicted F1 Improvement: {result['predicted_improvement_pct']:.2f}%")
    print(f"    Recommendation: {result['recommendation']}")
    
    # Save model
    print("\n[4/4] Saving trained model...")
    predictor.save('models/predictive_impact_model.pkl')
    
    print("\n" + "="*60)
    print("✓ DEMO COMPLETE")
    print("="*60)
    print("\nUsage in your pipeline:")
    print("  1. Detect quality issues → get quality_report")
    print("  2. Use predictor.predict(quality_report)")
    print("  3. Get recommendation: 'Worth fixing' or 'Skip'")
    print("  4. Only apply fixes if predicted improvement > 1%")
    print("\nBenefit: 100x faster than 5-fold CV!\n")
