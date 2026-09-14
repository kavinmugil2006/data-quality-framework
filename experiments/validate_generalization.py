"""
Generalization Validation Script
Test how well quality fixes generalize to unseen test data
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from quality_detector import QualityDetector
from quality_fixes import QualityFixer

class GeneralizationValidator:
    """Validate how quality fixes generalize to unseen data"""
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.results = {}
    
    def _prepare_data(self, X, y):
        """Encode categorical variables"""
        X = X.copy()
        
        # Fill missing numeric values with mean
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            X[col] = X[col].fillna(X[col].mean())
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = X[col].fillna('MISSING')
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Encode target
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = pd.Series(le_target.fit_transform(y.astype(str)), index=y.index)
        
        return X, y
    
    def test_generalization(self, df, dataset_name, target_col):
        """
        Test if quality fixes learned on train set generalize to test set
        
        Workflow:
        1. Split data into train (80%) and test (20%)
        2. Apply quality detection on train set
        3. Apply quality fixes learned from train to test
        4. Train models on fixed train data
        5. Evaluate on test data (unseen during fixing)
        """
        
        print(f"\n{'='*70}")
        print(f"GENERALIZATION TEST: {dataset_name}")
        print(f"{'='*70}")
        
        # Step 1: Split data
        print(f"\nDataset shape: {df.shape}")
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        df_train = pd.concat([X_train, y_train], axis=1)
        df_test = pd.concat([X_test, y_test], axis=1)
        
        print(f"Train set: {df_train.shape}")
        print(f"Test set:  {df_test.shape}")
        
        # Step 2: Detect quality issues on TRAIN set only
        print(f"\n[1/4] Detecting quality issues on TRAIN set...")
        detector = QualityDetector()
        train_quality_results = detector.detect_all(df_train)
        train_quality_score = detector.generate_quality_score(train_quality_results)
        print(f"Train Quality Score: {train_quality_score:.1f}/100")
        
        # Step 3: Apply fixes learned from TRAIN to both TRAIN and TEST
        print(f"\n[2/4] Applying fixes to TRAIN and TEST sets...")
        fixer = QualityFixer()
        fix_config = {
            'missing_values': 'drop',
            'duplicates': True,
            'outliers': False,
            'imbalance': False,
            'scale': False,
        }
        
        df_train_fixed = fixer.apply_fixes(df_train, fix_config)
        df_test_fixed = fixer.apply_fixes(df_test, fix_config)
        
        print(f"Train set after fix: {df_train_fixed.shape}")
        print(f"Test set after fix:  {df_test_fixed.shape}")
        
        # Step 4: Prepare data and train models
        print(f"\n[3/4] Training models on FIXED TRAIN set...")
        
        X_train_fixed = df_train_fixed.drop(columns=[target_col])
        y_train_fixed = df_train_fixed[target_col]
        
        X_test_fixed = df_test_fixed.drop(columns=[target_col])
        y_test_fixed = df_test_fixed[target_col]
        
        X_train_fixed, y_train_fixed = self._prepare_data(X_train_fixed, y_train_fixed)
        X_test_fixed, y_test_fixed = self._prepare_data(X_test_fixed, y_test_fixed)
        
        results = {}
        
        # Train and evaluate Random Forest
        print("\n  Evaluating Random Forest...")
        rf = RandomForestClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1)
        try:
            rf.fit(X_train_fixed, y_train_fixed)
            y_pred = rf.predict(X_test_fixed)
            
            f1 = f1_score(y_test_fixed, y_pred, zero_division=0, average='weighted')
            acc = accuracy_score(y_test_fixed, y_pred)
            
            results['RF'] = {'f1': f1, 'accuracy': acc}
            print(f"    F1 Score (Test): {f1:.4f}")
            print(f"    Accuracy (Test): {acc:.4f}")
        except Exception as e:
            print(f"    Error: {str(e)[:50]}")
            results['RF'] = {'f1': 0.0, 'accuracy': 0.0}
        
        # Train and evaluate Logistic Regression
        print("\n  Evaluating Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, random_state=self.random_state, n_jobs=-1)
        try:
            lr.fit(X_train_fixed, y_train_fixed)
            y_pred = lr.predict(X_test_fixed)
            
            f1 = f1_score(y_test_fixed, y_pred, zero_division=0, average='weighted')
            acc = accuracy_score(y_test_fixed, y_pred)
            
            results['LR'] = {'f1': f1, 'accuracy': acc}
            print(f"    F1 Score (Test): {f1:.4f}")
            print(f"    Accuracy (Test): {acc:.4f}")
        except Exception as e:
            print(f"    Error: {str(e)[:50]}")
            results['LR'] = {'f1': 0.0, 'accuracy': 0.0}
        
        # Step 5: Print generalization report
        print(f"\n[4/4] Generalization Analysis")
        print(f"\n{'='*70}")
        print(f"GENERALIZATION REPORT: {dataset_name}")
        print(f"{'='*70}")
        
        avg_f1 = np.mean([results[m]['f1'] for m in results.keys()])
        avg_acc = np.mean([results[m]['accuracy'] for m in results.keys()])
        
        print(f"\nTrain Quality Score: {train_quality_score:.1f}/100")
        print(f"Average Test F1:     {avg_f1:.4f}")
        print(f"Average Test Accuracy: {avg_acc:.4f}")
        
        print(f"\nPer-Model Test Performance:")
        for model, metrics in results.items():
            print(f"  {model}: F1={metrics['f1']:.4f}, Accuracy={metrics['accuracy']:.4f}")
        
        # Store results
        self.results[dataset_name] = {
            'train_quality_score': train_quality_score,
            'test_f1_scores': {m: results[m]['f1'] for m in results},
            'test_accuracy_scores': {m: results[m]['accuracy'] for m in results},
            'avg_test_f1': avg_f1,
            'avg_test_accuracy': avg_acc,
            'train_shape': df_train.shape,
            'test_shape': df_test.shape,
            'train_shape_fixed': df_train_fixed.shape,
            'test_shape_fixed': df_test_fixed.shape,
        }
    
    def save_validation_results(self):
        """Save validation results"""
        # Convert to serializable format
        results_serializable = {}
        for dataset, metrics in self.results.items():
            results_serializable[dataset] = {
                k: float(v) if isinstance(v, (int, float)) else v 
                for k, v in metrics.items()
            }
        
        with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'generalization_validation.json'), 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\n✓ Results saved to: generalization_validation.json")
    
    def generate_report(self):
        """Generate text report"""
        report = []
        report.append("\n" + "="*70)
        report.append("GENERALIZATION VALIDATION REPORT")
        report.append("="*70)
        
        for dataset, metrics in self.results.items():
            report.append(f"\n{dataset}:")
            report.append(f"  Train Quality Score: {metrics['train_quality_score']:.1f}/100")
            report.append(f"  Test F1 Score: {metrics['avg_test_f1']:.4f}")
            report.append(f"  Test Accuracy: {metrics['avg_test_accuracy']:.4f}")
            report.append(f"  Train shape (original -> fixed): {metrics['train_shape']} -> {metrics['train_shape_fixed']}")
            report.append(f"  Test shape (original -> fixed): {metrics['test_shape']} -> {metrics['test_shape_fixed']}")
        
        report.append("\n" + "="*70)
        
        report_text = "\n".join(report)
        
        with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'generalization_report.txt'), 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n✓ Report saved to: generalization_report.txt")


def load_datasets():
    """Load the same datasets as main experiments"""
    datasets = {}
    
    # Adult dataset (synthetic)
    print("\nLoading Adult dataset (synthetic)...")
    np.random.seed(42)
    n = 5000
    datasets['Adult'] = (pd.DataFrame({
        'age': np.random.randint(18, 80, n),
        'workclass': np.random.choice(['Private', 'Self-emp', 'Federal-gov', np.nan], n),
        'education': np.random.choice(['Bachelors', 'Masters', 'HS-grad', np.nan], n),
        'occupation': np.random.choice(['Tech', 'Sales', 'Admin', np.nan], n),
        'relationship': np.random.choice(['Husband', 'Wife', 'Child'], n),
        'hours-per-week': np.random.randint(1, 100, n),
        'income': np.random.choice(['<=50K', '>50K'], n, p=[0.76, 0.24])
    }), 'income')
    
    # Titanic dataset (synthetic)
    print("Loading Titanic dataset (synthetic)...")
    n = 891
    datasets['Titanic'] = (pd.DataFrame({
        'Pclass': np.random.choice([1, 2, 3], n, p=[0.16, 0.22, 0.62]),
        'Sex': np.random.choice(['male', 'female'], n, p=[0.65, 0.35]),
        'Age': np.random.normal(29, 15, n),
        'SibSp': np.random.poisson(0.5, n),
        'Parch': np.random.poisson(0.4, n),
        'Fare': np.random.gamma(2, 2, n) * 50,
        'Survived': np.random.choice([0, 1], n, p=[0.62, 0.38])
    }), 'Survived')
    
    # Credit Fraud dataset (synthetic)
    print("Loading Credit Fraud dataset (synthetic)...")
    n = 10000
    datasets['Credit Fraud'] = (pd.DataFrame({
        **{f'V{i}': np.random.randn(n) for i in range(1, 29)},
        'Amount': np.abs(np.random.randn(n) * 100),
        'Class': np.random.choice([0, 1], n, p=[0.998, 0.002])
    }), 'Class')
    
    return datasets


def main():
    """Run generalization validation"""
    print("\n" + "="*70)
    print("GENERALIZATION VALIDATION: Test on Unseen Data")
    print("="*70)
    
    validator = GeneralizationValidator()
    datasets = load_datasets()
    
    for dataset_name, (df, target_col) in datasets.items():
        try:
            validator.test_generalization(df, dataset_name, target_col)
        except Exception as e:
            print(f"Error testing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    validator.save_validation_results()
    validator.generate_report()
    
    print("\n✓ All generalization tests completed!")


if __name__ == '__main__':
    main()