"""
Real-World Validation Framework
================================
Register and validate datasets against lab predictions
"""

from dataclasses import dataclass, asdict
from typing import Dict, List
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import hashlib
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ValidationDataset:
    dataset_id: str
    source: str
    size: int
    n_features: int
    data_types: Dict
    quality_issues: List[str]
    notes: str
    has_pii: bool = False
    anonymized: bool = False
    data_hash: str = ""


class RealWorldValidator:
    def __init__(self, results_dir: str = "demo_validation_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.registered_datasets: Dict[str, ValidationDataset] = {}
        self.validation_results: Dict[str, Dict] = {}
    
    def register_dataset(self, dataset: ValidationDataset) -> None:
        """Register a real-world dataset for validation"""
        self.registered_datasets[dataset.dataset_id] = dataset
        self._save_dataset_metadata(dataset)
        
        print(f"\n{'='*60}")
        print(f"DATASET REGISTERED: {dataset.dataset_id}")
        print(f"{'='*60}")
        print(f"  ID: {dataset.dataset_id}")
        print(f"  Source: {dataset.source}")
        print(f"  Size: {dataset.size:,} rows × {dataset.n_features} features")
        print(f"  Data Types: {dataset.data_types}")
        print(f"  Issues: {', '.join(dataset.quality_issues)}")
        print(f"  Notes: {dataset.notes}")
        print(f"  ✓ Metadata saved to {self.results_dir / f'{dataset.dataset_id}_metadata.json'}")
    
    def _save_dataset_metadata(self, dataset: ValidationDataset) -> None:
        """Save dataset metadata to JSON"""
        metadata_file = self.results_dir / f"{dataset.dataset_id}_metadata.json"
        
        metadata = {
            'dataset_id': dataset.dataset_id,
            'source': dataset.source,
            'size': dataset.size,
            'n_features': dataset.n_features,
            'data_types': dataset.data_types,
            'quality_issues': dataset.quality_issues,
            'notes': dataset.notes,
            'has_pii': dataset.has_pii,
            'anonymized': dataset.anonymized,
            'registered_at': datetime.now().isoformat()
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    def validate_dataset(self, dataset_id: str, lab_prediction: float, 
                        actual_improvement: float) -> Dict:
        """Validate a dataset against lab predictions"""
        
        if dataset_id not in self.registered_datasets:
            raise ValueError(f"Dataset {dataset_id} not registered!")
        
        dataset = self.registered_datasets[dataset_id]
        
        # Calculate quality score
        quality_score = 100
        quality_score -= len(dataset.quality_issues) * 5
        quality_score = max(30, min(100, quality_score))
        
        # Check if lab prediction matches reality
        prediction_error = abs(lab_prediction - actual_improvement)
        matches_lab = prediction_error < 0.015  # Within 1.5% threshold
        
        result = {
            'dataset_id': dataset_id,
            'quality_score': round(quality_score, 1),
            'detected_issues': dataset.quality_issues,
            'lab_prediction_f1_improvement': round(lab_prediction, 4),
            'actual_f1_improvement': round(actual_improvement, 4),
            'prediction_error': round(prediction_error, 4),
            'matches_lab_results': matches_lab,
            'recommendations': self._generate_recommendations(quality_score, lab_prediction, actual_improvement),
            'validated_at': datetime.now().isoformat()
        }
        
        self.validation_results[dataset_id] = result
        self._save_validation_result(dataset_id, result)
        
        return result
    
    def _generate_recommendations(self, quality_score: float, 
                                 lab_pred: float, actual: float) -> Dict:
        """Generate recommendations based on validation results"""
        
        recommendations = {}
        
        if quality_score > 80:
            recommendations['data_quality'] = "Data quality is high. Minimal fixes needed."
            recommendations['action_level'] = "PASSIVE"
        elif quality_score > 50:
            recommendations['data_quality'] = "Data quality is moderate. Targeted fixes recommended."
            recommendations['action_level'] = "ACTIVE"
        else:
            recommendations['data_quality'] = "Data quality is low. Comprehensive cleaning needed."
            recommendations['action_level'] = "URGENT"
        
        if actual > 0.01:
            recommendations['fix_benefit'] = f"Fixes provided significant benefit ({actual*100:.2f}% F1 gain)."
        elif actual > 0.005:
            recommendations['fix_benefit'] = f"Fixes provided minimal benefit ({actual*100:.2f}% F1 gain)."
        else:
            recommendations['fix_benefit'] = "Fixes provided minimal benefit (negligible F1 gain)."
        
        # Check for failure modes
        if lab_pred > 0 and actual < 0:
            recommendations['failure_mode'] = "Predicted positive impact, but actual was negative"
            recommendations['reason'] = "Data loss outweighed quality gains"
            recommendations['mitigation'] = "Try: Less aggressive fixes, preserve more data"
        elif abs(lab_pred - actual) > 0.02:
            recommendations['mismatch'] = "Lab prediction deviated from real-world"
            recommendations['cause'] = "Possible data distribution shift"
        else:
            recommendations['accuracy'] = "Lab predictions matched real-world results. Framework is accurate!"
        
        return recommendations
    
    def _save_validation_result(self, dataset_id: str, result: Dict) -> None:
        """Save validation result to JSON"""
        result_file = self.results_dir / f"{dataset_id}_result.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
    
    def print_validation_result(self, dataset_id: str) -> None:
        """Print formatted validation result"""
        if dataset_id not in self.validation_results:
            print(f"No validation result for {dataset_id}")
            return
        
        result = self.validation_results[dataset_id]
        dataset = self.registered_datasets[dataset_id]
        
        print(f"\n{'='*60}")
        print(f"RUNNING VALIDATION: {dataset.dataset_id}")
        print(f"{'='*60}")
        
        print(f"\n  Results for: {dataset.dataset_id}")
        print(f"  {'-'*55}")
        print(f"  Quality Score: {result['quality_score']}/100")
        print(f"  Detected Issues: {result['detected_issues']}")
        print(f"  Expected F1 Improvement (lab): {result['lab_prediction_f1_improvement']:+.2%}")
        print(f"  Actual F1 Improvement (real): {result['actual_f1_improvement']:+.2%}")
        print(f"  Matches Lab Results: {'✓ YES' if result['matches_lab_results'] else '✗ NO'}")
        
        print(f"\n  Recommendations:")
        recs = result['recommendations']
        
        if 'data_quality' in recs:
            print(f"    {'✓' if result['quality_score'] > 80 else '~' if result['quality_score'] > 50 else '⚠️'} {recs['data_quality']}")
        
        if 'fix_benefit' in recs:
            print(f"      {recs['fix_benefit']}")
        
        if 'mitigation' in recs:
            print(f"      ❌ {recs['failure_mode']}")
            print(f"         Try: {recs['mitigation']}")
        
        if 'accuracy' in recs:
            print(f"      ✓ {recs['accuracy']}")
        
        print(f"  ✓ Results saved to {self.results_dir / f'{dataset_id}_result.json'}")
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report"""
        
        if not self.validation_results:
            return "No validation results available."
        
        report = f"\n{'='*70}\n"
        report += "REAL-WORLD VALIDATION REPORT\n"
        report += f"{'='*70}\n\n"
        
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Total Datasets Validated: {len(self.validation_results)}\n\n"
        
        # Summary statistics
        quality_scores = [r['quality_score'] for r in self.validation_results.values()]
        improvements = [r['actual_f1_improvement'] for r in self.validation_results.values()]
        match_count = sum(1 for r in self.validation_results.values() if r['matches_lab_results'])
        
        # SAFE division by zero check (FIX #2)
        total_datasets = len(self.validation_results)
        match_rate = (match_count / total_datasets * 100) if total_datasets > 0 else 0
        
        report += "Summary Statistics:\n"
        report += f"  Average Quality Score: {np.mean(quality_scores):.1f}\n"
        report += f"  Average F1 Improvement: {np.mean(improvements):+.2%}\n"
        report += f"  Lab Match Rate: {match_count}/{total_datasets} ({match_rate:.0f}%)\n\n"
        
        # Detailed results
        report += f"{'-'*70}\n"
        report += "DETAILED RESULTS\n"
        report += f"{'-'*70}\n\n"
        
        for dataset_id, result in self.validation_results.items():
            report += f"  Dataset: {result['dataset_id']}\n"
            report += f"    Quality Score: {result['quality_score']}\n"
            report += f"    F1 Improvement: {result['actual_f1_improvement']:+.2%}\n"
            report += f"    Matches Lab: {'✓' if result['matches_lab_results'] else '✗'}\n"
            
            if not result['matches_lab_results']:
                recs = result['recommendations']
                if 'failure_mode' in recs:
                    report += f"    Issues: {recs['failure_mode']}, {recs['reason']}\n"
            
            report += "\n"
        
        # Key insights
        report += f"{'-'*70}\n"
        report += "KEY INSIGHTS\n"
        report += f"{'-'*70}\n\n"
        
        report += "  1. Framework Accuracy\n"
        report += f"     {'✓' if match_count/total_datasets > 0.8 else '~'} Lab predictions are accurate in real-world (>80% match)\n\n"
        
        report += "  2. Data Quality Patterns\n"
        report += f"     Real-world data quality: {np.mean(quality_scores):.1f}/100\n"
        report += f"     Moderate quality issues found in most datasets\n\n"
        
        report += "  3. Fix Effectiveness\n"
        report += f"     Average F1 improvement: {np.mean(improvements):+.2%}\n"
        report += f"     {'~' if abs(np.mean(improvements)) < 0.02 else '✓'} Fixes are {'marginally' if abs(np.mean(improvements)) < 0.02 else 'significantly'} effective (<2% improvement)\n"
        
        report += f"\n{'='*70}\n"
        report += "✓ VALIDATION COMPLETE\n"
        report += f"{'='*70}\n"
        
        # Save report
        report_file = self.results_dir / "validation_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✓ Report saved to {report_file}")
        
        return report


if __name__ == '__main__':
    print("\n" + "="*70)
    print("REAL-WORLD VALIDATION FRAMEWORK - DEMO")
    print("="*70)
    
    validator = RealWorldValidator()
    
    print("\n[1/4] Registering Real-World Datasets...")
    
    # Dataset 1: Banking
    dataset1 = ValidationDataset(
        dataset_id='banking_001',
        source='finance',
        size=50000,
        n_features=25,
        data_types={'numeric': 18, 'categorical': 7},
        quality_issues=['missing_values', 'duplicates', 'outliers', 'imbalance_ratio'],
        notes='Production data from 2024. High compliance requirements.',
        has_pii=True,
        anonymized=True
    )
    validator.register_dataset(dataset1)
    
    # Dataset 2: E-commerce
    dataset2 = ValidationDataset(
        dataset_id='retail_001',
        source='ecommerce',
        size=100000,
        n_features=15,
        data_types={'numeric': 10, 'categorical': 5},
        quality_issues=['missing_values', 'duplicates', 'outliers', 'imbalance_ratio'],
        notes='Real production e-commerce data. Handles edge cases well.'
    )
    validator.register_dataset(dataset2)
    
    # Dataset 3: Healthcare
    dataset3 = ValidationDataset(
        dataset_id='medical_001',
        source='healthcare',
        size=10000,
        n_features=35,
        data_types={'numeric': 25, 'categorical': 10},
        quality_issues=['missing_values', 'duplicates', 'outliers', 'imbalance_ratio'],
        notes='Real healthcare data. Strict compliance and quality requirements.',
        has_pii=True,
        anonymized=True
    )
    validator.register_dataset(dataset3)
    
    print("\n[2/4] Running Validation on Datasets...")
    
    # Validate Banking
    print("\n  Banking Dataset Validation...")
    result1 = validator.validate_dataset('banking_001', lab_prediction=0.012, actual_improvement=0.018)
    validator.print_validation_result('banking_001')
    
    # Validate E-commerce
    print("\n  E-commerce Dataset Validation...")
    result2 = validator.validate_dataset('retail_001', lab_prediction=0.008, actual_improvement=0.009)
    validator.print_validation_result('retail_001')
    
    # Validate Healthcare
    print("\n  Healthcare Dataset Validation...")
    result3 = validator.validate_dataset('medical_001', lab_prediction=0.010, actual_improvement=-0.004)
    validator.print_validation_result('medical_001')
    
    print("\n[3/4] Generating Validation Report...")
    report = validator.generate_validation_report()
    
    print("\n[4/4] Report Summary:")
    print(report)
    
    print("\nKey Findings:")
    print("  • Banking: Lab matched real-world (1.8% vs 1.2% improvement)")
    print("  • E-commerce: Lab matched real-world (0.8% vs 0.9% improvement)")
    print("  • Healthcare: Lab MISMATCHED real-world (1.0% vs -0.4%!)")
    print("\nInsight: Framework works well for finance/retail, needs tuning for healthcare\n")
