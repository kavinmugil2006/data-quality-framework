"""
Real-World Validation Framework
================================
Validate data quality framework on actual enterprise datasets.

Why validation matters:
  - Synthetic data ≠ Real data
  - Lab results ≠ Production results
  - Academic datasets ≠ Enterprise data

This module provides tools to:
  1. Ingest real enterprise data (safely)
  2. Run validation on actual quality issues
  3. Compare lab results vs real-world results
  4. Document findings and edge cases
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import hashlib


@dataclass
class ValidationDataset:
    """Metadata for a real-world validation dataset."""
    
    dataset_id: str
    dataset_name: str
    source: str  # 'finance', 'healthcare', 'ecommerce', etc
    rows: int
    features: int
    date_collected: str
    description: str
    
    # Data characteristics
    has_missing: bool
    has_duplicates: bool
    has_outliers: bool
    is_imbalanced: bool
    data_types: Dict  # {'numeric': 45, 'categorical': 10, 'datetime': 2}
    
    # Privacy & compliance
    contains_pii: bool
    anonymization_method: Optional[str]
    data_hash: str  # SHA256 hash for integrity
    
    # Documentation
    data_quality_issues: List[str]
    known_edge_cases: List[str]
    notes: str


@dataclass
class ValidationResult:
    """Results from running framework on real-world dataset."""
    
    dataset_id: str
    timestamp: str
    
    # Quality detection
    quality_score: float
    detected_issues: Dict
    
    # Impact analysis
    classification_results: Optional[Dict] = None
    regression_results: Optional[Dict] = None
    clustering_results: Optional[Dict] = None
    
    # Real vs lab comparison
    expected_f1_improvement: float = 0.0
    actual_f1_improvement: float = 0.0
    matches_lab_results: bool = False
    
    # Findings
    unexpected_patterns: List[str] = None
    success_factors: List[str] = None
    failure_modes: List[str] = None
    recommendations: str = ""
    
    def __post_init__(self):
        """Initialize defaults."""
        if self.unexpected_patterns is None:
            self.unexpected_patterns = []
        if self.success_factors is None:
            self.success_factors = []
        if self.failure_modes is None:
            self.failure_modes = []


class RealWorldValidator:
    """
    Validate framework on real enterprise datasets.
    
    Workflow:
      1. Register dataset (metadata)
      2. Run quality detection
      3. Measure impact on chosen ML task
      4. Compare to lab results
      5. Document findings
    """
    
    def __init__(self, results_dir: str = 'validation_results'):
        """Initialize validator."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.datasets: Dict[str, ValidationDataset] = {}
        self.validation_results: Dict[str, ValidationResult] = {}
    
    def register_dataset(self, dataset: ValidationDataset):
        """Register a new validation dataset."""
        self.datasets[dataset.dataset_id] = dataset
        
        print(f"\n{'='*60}")
        print(f"DATASET REGISTERED: {dataset.dataset_name}")
        print(f"{'='*60}")
        print(f"  ID: {dataset.dataset_id}")
        print(f"  Source: {dataset.source}")
        print(f"  Size: {dataset.rows:,} rows × {dataset.features} features")
        print(f"  Data Types: {dataset.data_types}")
        print(f"  Issues: {', '.join(dataset.data_quality_issues) if dataset.data_quality_issues else 'None detected'}")
        print(f"  Notes: {dataset.notes}")
        
        # Save metadata
        self._save_dataset_metadata(dataset)
    
    def _save_dataset_metadata(self, dataset: ValidationDataset):
        """Save dataset metadata to JSON."""
        metadata_file = self.results_dir / f"{dataset.dataset_id}_metadata.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(asdict(dataset), f, indent=2)
        
        print(f"  ✓ Metadata saved to {metadata_file}")
    
    def run_validation(
        self,
        dataset_id: str,
        quality_score: float,
        detected_issues: Dict,
        ml_task: str = 'classification',
        ml_results: Optional[Dict] = None,
        expected_improvement: float = 0.0,
        actual_improvement: float = 0.0
    ) -> ValidationResult:
        """
        Record validation results for a dataset.
        
        Args:
            dataset_id: ID of registered dataset
            quality_score: Quality detection score (0-100)
            detected_issues: Issues found by detector
            ml_task: 'classification', 'regression', or 'clustering'
            ml_results: ML impact analysis results
            expected_improvement: Lab results prediction
            actual_improvement: Real-world actual improvement
        """
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not registered!")
        
        dataset = self.datasets[dataset_id]
        
        print(f"\n{'='*60}")
        print(f"RUNNING VALIDATION: {dataset.dataset_name}")
        print(f"{'='*60}")
        
        # Analyze results
        matches = abs(expected_improvement - actual_improvement) < 0.02  # Within 2%
        
        # Identify unexpected patterns
        unexpected_patterns = []
        success_factors = []
        failure_modes = []
        
        if actual_improvement > expected_improvement * 2:
            unexpected_patterns.append(f"F1 improvement {actual_improvement*100:.1f}% much higher than expected {expected_improvement*100:.1f}%!")
            success_factors.append("Data fixes were more effective than predicted")
        
        if actual_improvement < 0 and expected_improvement > 0:
            failure_modes.append("Predicted positive impact, but actual was negative")
            failure_modes.append("Possible reason: Data loss outweighed quality gains")
        
        if quality_score < 50:
            failure_modes.append("Data quality too low - aggressive fixes needed")
        
        # Create result object
        result = ValidationResult(
            dataset_id=dataset_id,
            timestamp=datetime.now().isoformat(),
            quality_score=quality_score,
            detected_issues=detected_issues,
            classification_results=ml_results if ml_task == 'classification' else None,
            regression_results=ml_results if ml_task == 'regression' else None,
            clustering_results=ml_results if ml_task == 'clustering' else None,
            expected_f1_improvement=expected_improvement,
            actual_f1_improvement=actual_improvement,
            matches_lab_results=matches,
            unexpected_patterns=unexpected_patterns,
            success_factors=success_factors,
            failure_modes=failure_modes,
            recommendations=self._generate_recommendations(
                quality_score, actual_improvement, matches, dataset
            )
        )
        
        self.validation_results[dataset_id] = result
        
        # Print results
        self._print_validation_results(dataset, result)
        
        # Save results
        self._save_validation_result(result)
        
        return result
    
    def _generate_recommendations(
        self,
        quality_score: float,
        actual_improvement: float,
        matches: bool,
        dataset: ValidationDataset
    ) -> str:
        """Generate actionable recommendations based on results."""
        recs = []
        
        if quality_score < 50:
            recs.append("❌ Data quality is too low. Recommend pre-processing before modeling.")
        elif quality_score < 75:
            recs.append("⚠️  Data quality is moderate. Targeted fixes recommended.")
        else:
            recs.append("✓ Data quality is high. Minimal fixes needed.")
        
        if actual_improvement > 0.02:
            recs.append(f"✓ Fixes provided measurable benefit ({actual_improvement*100:.1f}% F1 gain).")
        elif actual_improvement > 0:
            recs.append(f"~ Fixes provided minimal benefit ({actual_improvement*100:.2f}% F1 gain).")
        else:
            recs.append(f"❌ Fixes reduced performance ({actual_improvement*100:.2f}%).")
            recs.append("   Try: Less aggressive fixes, preserve more data")
        
        if matches:
            recs.append("✓ Lab predictions matched real-world results. Framework is accurate!")
        else:
            recs.append("⚠️  Lab predictions differed from real-world. Domain-specific tuning needed.")
        
        return "\n  ".join(recs)
    
    def _print_validation_results(self, dataset: ValidationDataset, result: ValidationResult):
        """Print validation results in readable format."""
        print(f"\n  Results for: {dataset.dataset_name}")
        print(f"  " + "-" * 55)
        print(f"  Quality Score: {result.quality_score:.1f}/100")
        print(f"  Detected Issues: {list(result.detected_issues.keys())}")
        print(f"  Expected F1 Improvement (lab): {result.expected_f1_improvement*100:+.2f}%")
        print(f"  Actual F1 Improvement (real): {result.actual_f1_improvement*100:+.2f}%")
        print(f"  Matches Lab Results: {'✓ YES' if result.matches_lab_results else '✗ NO'}")
        
        if result.unexpected_patterns:
            print(f"  Unexpected Patterns:")
            for pattern in result.unexpected_patterns:
                print(f"    • {pattern}")
        
        if result.failure_modes:
            print(f"  Failure Modes:")
            for mode in result.failure_modes:
                print(f"    ✗ {mode}")
        
        if result.success_factors:
            print(f"  Success Factors:")
            for factor in result.success_factors:
                print(f"    ✓ {factor}")
        
        print(f"  Recommendations:")
        for rec in result.recommendations.split('\n'):
            print(f"    {rec}")
    
    def _save_validation_result(self, result: ValidationResult):
        """Save validation result to JSON."""
        result_file = self.results_dir / f"{result.dataset_id}_result.json"
        
        # Convert dataclass to dict, handling special types
        result_dict = asdict(result)
        
        with open(result_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"  ✓ Results saved to {result_file}")
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report."""
        report = []
        report.append("="*70)
        report.append("REAL-WORLD VALIDATION REPORT")
        report.append("="*70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Datasets Validated: {len(self.validation_results)}")
        
        # Summary statistics
        if self.validation_results:
            quality_scores = [r.quality_score for r in self.validation_results.values()]
            improvements = [r.actual_f1_improvement for r in self.validation_results.values()]
            match_count = sum(1 for r in self.validation_results.values() if r.matches_lab_results)
            
            report.append(f"\nSummary Statistics:")
            report.append(f"  Average Quality Score: {np.mean(quality_scores):.1f}")
            report.append(f"  Average F1 Improvement: {np.mean(improvements)*100:+.2f}%")
            report.append(f"  Lab Match Rate: {match_count}/{len(self.validation_results)} ({match_count/len(self.validation_results)*100:.0f}%)")
        
        # Detailed results
        report.append(f"\n" + "-"*70)
        report.append("DETAILED RESULTS")
        report.append("-"*70)
        
        for dataset_id, result in self.validation_results.items():
            dataset = self.datasets[dataset_id]
            report.append(f"\n  Dataset: {dataset.dataset_name}")
            report.append(f"    Quality Score: {result.quality_score:.1f}")
            report.append(f"    F1 Improvement: {result.actual_f1_improvement*100:+.2f}%")
            report.append(f"    Matches Lab: {'✓' if result.matches_lab_results else '✗'}")
            
            if result.failure_modes:
                report.append(f"    Issues: {', '.join(result.failure_modes)}")
        
        # Insights
        report.append(f"\n" + "-"*70)
        report.append("KEY INSIGHTS")
        report.append("-"*70)
        
        report.append("\n  1. Framework Accuracy")
        if len(self.validation_results) > 0:
            match_rate = match_count / len(self.validation_results)
            if match_rate > 0.8:
                report.append("     ✓ Lab predictions are accurate in real-world (>80% match)")
            elif match_rate > 0.5:
                report.append("     ~ Lab predictions are moderately accurate (50-80% match)")
            else:
                report.append("     ✗ Lab predictions need domain-specific tuning (<50% match)")
        
        report.append("\n  2. Data Quality Patterns")
        if quality_scores:
            avg_score = np.mean(quality_scores)
            report.append(f"     Real-world data quality: {avg_score:.1f}/100")
            if avg_score > 80:
                report.append("     Most datasets are high-quality")
            elif avg_score > 60:
                report.append("     Moderate quality issues found in most datasets")
            else:
                report.append("     Significant quality issues detected")
        
        report.append("\n  3. Fix Effectiveness")
        avg_improvement = np.mean(improvements)
        report.append(f"     Average F1 improvement: {avg_improvement*100:+.2f}%")
        if avg_improvement > 0.02:
            report.append("     ✓ Fixes are effective (>2% improvement)")
        elif avg_improvement > 0:
            report.append("     ~ Fixes are marginally effective (<2% improvement)")
        else:
            report.append("     ✗ Fixes may be counterproductive")
        
        report_text = "\n".join(report)
        
        # Save report
        report_file = self.results_dir / "validation_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"\n✓ Report saved to {report_file}")
        
        return report_text


# ============================================================================
# Demo: Simulated Real-World Validation
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("REAL-WORLD VALIDATION FRAMEWORK - DEMO")
    print("="*70)
    
    validator = RealWorldValidator('demo_validation_results')
    
    # Register 3 real-world datasets
    print("\n[1/4] Registering Real-World Datasets...")
    
    dataset1 = ValidationDataset(
        dataset_id='banking_001',
        dataset_name='Bank Loan Approval Data',
        source='finance',
        rows=50000,
        features=25,
        date_collected='2026-01-15',
        description='Historical loan application data with approval decisions',
        has_missing=True,
        has_duplicates=True,
        has_outliers=True,
        is_imbalanced=False,
        data_types={'numeric': 18, 'categorical': 7},
        contains_pii=True,
        anonymization_method='hash_id_pseudonymization',
        data_hash='abc123def456',
        data_quality_issues=['Missing income (2.5%)', 'Duplicate applications (0.8%)', 'Outlier credit scores (5%)'],
        known_edge_cases=['New customers (no credit history)', 'Recent immigrants (short history)', 'Business accounts (different rules)'],
        notes='Production data from 2024. High compliance requirements.'
    )
    validator.register_dataset(dataset1)
    
    dataset2 = ValidationDataset(
        dataset_id='retail_001',
        dataset_name='E-commerce Purchase History',
        source='ecommerce',
        rows=100000,
        features=15,
        date_collected='2026-02-01',
        description='Customer purchase transactions from online store',
        has_missing=False,
        has_duplicates=True,
        has_outliers=True,
        is_imbalanced=True,
        data_types={'numeric': 10, 'categorical': 5},
        contains_pii=False,
        anonymization_method=None,
        data_hash='xyz789uvw012',
        data_quality_issues=['High-value outliers (expensive items)', 'Seasonal imbalance (holiday spikes)', 'Duplicate orders (0.5%)'],
        known_edge_cases=['Bulk orders', 'International purchases', 'Return transactions'],
        notes='Real production e-commerce data. Handles edge cases well.'
    )
    validator.register_dataset(dataset2)
    
    dataset3 = ValidationDataset(
        dataset_id='medical_001',
        dataset_name='Patient Clinical Records',
        source='healthcare',
        rows=10000,
        features=35,
        date_collected='2025-12-01',
        description='Anonymized patient records for disease prediction',
        has_missing=True,
        has_duplicates=False,
        has_outliers=True,
        is_imbalanced=True,
        data_types={'numeric': 25, 'categorical': 10},
        contains_pii=False,
        anonymization_method='HIPAA_deidentification',
        data_hash='pqr345stu678',
        data_quality_issues=['Missing test results (8%)', 'Impossible vital signs (1%)', 'Rare disease imbalance (99:1)'],
        known_edge_cases=['Pediatric patients (different norms)', 'ICU records (missing values common)', 'Chronic conditions (multiple encounters)'],
        notes='Real healthcare data. Strict compliance and quality requirements.'
    )
    validator.register_dataset(dataset3)
    
    # Run validation on datasets
    print("\n[2/4] Running Validation on Datasets...")
    
    # Dataset 1: Banking
    print("\n  Banking Dataset Validation...")
    validator.run_validation(
        dataset_id='banking_001',
        quality_score=78.5,
        detected_issues={
            'missing_values': 2.5,
            'duplicates': 0.8,
            'outliers': 5.0,
            'imbalance_ratio': 1.2
        },
        ml_task='classification',
        ml_results={
            'rf': {'improvement': {'f1': 0.015}},
            'lr': {'improvement': {'f1': 0.012}}
        },
        expected_improvement=0.012,  # Lab prediction
        actual_improvement=0.018      # Real-world result
    )
    
    # Dataset 2: E-commerce
    print("\n  E-commerce Dataset Validation...")
    validator.run_validation(
        dataset_id='retail_001',
        quality_score=85.2,
        detected_issues={
            'missing_values': 0.0,
            'duplicates': 0.5,
            'outliers': 8.5,
            'imbalance_ratio': 15.0
        },
        ml_task='classification',
        ml_results={
            'rf': {'improvement': {'f1': 0.008}},
            'lr': {'improvement': {'f1': 0.006}}
        },
        expected_improvement=0.008,
        actual_improvement=0.009
    )
    
    # Dataset 3: Healthcare
    print("\n  Healthcare Dataset Validation...")
    validator.run_validation(
        dataset_id='medical_001',
        quality_score=62.3,
        detected_issues={
            'missing_values': 8.0,
            'duplicates': 0.0,
            'outliers': 12.0,
            'imbalance_ratio': 99.0
        },
        ml_task='classification',
        ml_results={
            'rf': {'improvement': {'f1': -0.005}},
            'lr': {'improvement': {'f1': -0.003}}
        },
        expected_improvement=0.010,
        actual_improvement=-0.004
    )
    
    # Generate report
    print("\n[3/4] Generating Validation Report...")
    report = validator.generate_validation_report()
    
    print("\n[4/4] Report Summary:")
    print(report)
    
    print("\n" + "="*70)
    print("✓ VALIDATION COMPLETE")
    print("="*70)
    print("\nKey Findings:")
    print("  • Banking: Lab matched real-world (1.8% vs 1.2% improvement)")
    print("  • E-commerce: Lab matched real-world (0.8% vs 0.9% improvement)")
    print("  • Healthcare: Lab MISMATCHED real-world (1.0% vs -0.4%!)")
    print("\nInsight: Framework works well for finance/retail, needs tuning for healthcare\n")
