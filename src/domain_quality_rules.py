"""
Domain-Specific Quality Rules Module
====================================
Different domains have different quality requirements.

Example:
  Financial data: Strict rules on missing values (regulatory requirement)
  Healthcare data: Strict rules on outliers (impossible values harm diagnosis)
  E-commerce data: Looser rules on outliers (very expensive orders are rare but real)

This module lets you define custom quality thresholds per domain.
"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class DomainQualityRules:
    """Quality rules for a specific domain."""
    
    domain_name: str
    description: str
    
    # Completeness rules (missing values)
    max_missing_pct: float = 5.0  # Allow up to 5% missing
    critical_missing_columns: List[str] = None  # Columns that can't be missing
    
    # Consistency rules (duplicates)
    max_duplicates_pct: float = 2.0
    allow_exact_duplicates: bool = False  # Some domains allow exact duplicates
    
    # Validity rules (outliers)
    outlier_detection_method: str = 'iqr'  # 'iqr', 'zscore', 'isolation_forest'
    outlier_threshold_iqr: float = 3.0  # 3.0 × IQR = conservative
    outlier_threshold_zscore: float = 3.5
    flag_but_dont_remove: bool = False  # Flag for review instead of removal
    
    # Class balance rules
    max_imbalance_ratio: float = 10.0  # Allow up to 10:1 imbalance
    critical_minority_size: int = 50  # Minimum samples for minority class
    
    # Feature correlation rules
    max_correlation: float = 0.95  # Flag highly correlated features
    allow_multicollinearity: bool = False
    
    # Type consistency rules
    strict_type_checking: bool = False  # Some domains require strict types
    
    # Data freshness rules (time-series data)
    allow_timestamp_gaps: bool = False
    max_timestamp_gap_days: Optional[int] = None
    
    # Domain-specific rules
    regulatory_compliance: Optional[Dict] = None  # Financial: PII rules, etc
    medical_validity_rules: Optional[Dict] = None  # Healthcare: impossible values
    
    def __post_init__(self):
        """Initialize default lists."""
        if self.critical_missing_columns is None:
            self.critical_missing_columns = []
        if self.regulatory_compliance is None:
            self.regulatory_compliance = {}
        if self.medical_validity_rules is None:
            self.medical_validity_rules = {}
    
    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)


# ============================================================================
# Pre-defined Domain Rules
# ============================================================================

FINANCE_RULES = DomainQualityRules(
    domain_name='Finance',
    description='Strict rules for financial data (regulatory compliance, risk management)',
    
    max_missing_pct=0.1,  # Very strict: <0.1% missing (regulatory requirement)
    critical_missing_columns=['transaction_amount', 'account_id', 'date', 'institution_code'],
    
    max_duplicates_pct=0.5,  # Very strict on duplicates (fraud risk)
    allow_exact_duplicates=False,
    
    outlier_detection_method='zscore',
    outlier_threshold_zscore=4.0,  # More conservative for financial outliers
    flag_but_dont_remove=True,  # Flag for manual review (might be real fraud patterns)
    
    max_imbalance_ratio=100.0,  # Allow extreme imbalance (fraud is rare!)
    critical_minority_size=10,
    
    max_correlation=0.90,
    allow_multicollinearity=False,
    strict_type_checking=True,
    
    regulatory_compliance={
        'require_pii_encryption': True,
        'require_audit_trail': True,
        'max_data_retention_days': 2555,  # ~7 years
        'require_transaction_id': True,
    }
)

HEALTHCARE_RULES = DomainQualityRules(
    domain_name='Healthcare',
    description='Strict rules for healthcare data (patient safety, medical validity)',
    
    max_missing_pct=1.0,  # Strict but allow some missing (patient data is complex)
    critical_missing_columns=['patient_id', 'diagnosis', 'treatment_date'],
    
    max_duplicates_pct=0.1,  # Very strict (duplicate records = dosage errors)
    allow_exact_duplicates=False,
    
    outlier_detection_method='isolation_forest',  # Better for clinical outliers
    flag_but_dont_remove=True,  # Always flag for doctor review
    
    max_imbalance_ratio=20.0,
    critical_minority_size=50,  # Need enough rare disease cases
    
    strict_type_checking=True,
    
    medical_validity_rules={
        'age_min': 0,
        'age_max': 150,
        'hemoglobin_min': 5.0,
        'hemoglobin_max': 20.0,
        'blood_pressure_systolic_min': 40,
        'blood_pressure_systolic_max': 250,
        'glucose_min': 40,
        'glucose_max': 600,
        'require_clinical_notes': False,
    }
)

ECOMMERCE_RULES = DomainQualityRules(
    domain_name='E-commerce',
    description='Looser rules for e-commerce (allows rare expensive items, edge cases)',
    
    max_missing_pct=3.0,  # Allow some missing (product pages may have incomplete info)
    critical_missing_columns=['product_id', 'price', 'purchase_date'],
    
    max_duplicates_pct=1.0,
    allow_exact_duplicates=True,  # Same customer buying same item twice
    
    outlier_detection_method='iqr',
    outlier_threshold_iqr=1.5,  # Less strict (expensive items are real)
    flag_but_dont_remove=False,  # Can remove outliers safely
    
    max_imbalance_ratio=50.0,
    critical_minority_size=20,
    
    max_correlation=0.95,
    allow_multicollinearity=False,
    strict_type_checking=False,  # Allow flexible types (user-generated content)
)

SENTIMENT_ANALYSIS_RULES = DomainQualityRules(
    domain_name='Sentiment Analysis',
    description='Rules for NLP/sentiment data (text classification)',
    
    max_missing_pct=2.0,
    critical_missing_columns=['text', 'label'],
    
    max_duplicates_pct=5.0,  # Allow some duplicates (same sentiment expressed twice)
    allow_exact_duplicates=True,
    
    outlier_detection_method='isolation_forest',
    flag_but_dont_remove=True,  # Unusual text might be sarcasm/irony
    
    max_imbalance_ratio=10.0,
    critical_minority_size=100,
    
    strict_type_checking=False,
)

IOT_SENSOR_RULES = DomainQualityRules(
    domain_name='IoT Sensors',
    description='Rules for time-series sensor data (temperature, pressure, etc)',
    
    max_missing_pct=5.0,
    critical_missing_columns=['timestamp', 'sensor_id', 'measurement'],
    
    max_duplicates_pct=0.5,  # Strict on duplicates (sensor drift indicator)
    allow_exact_duplicates=False,
    
    outlier_detection_method='zscore',
    outlier_threshold_zscore=3.0,
    flag_but_dont_remove=False,
    
    allow_timestamp_gaps=False,
    max_timestamp_gap_days=0,  # No gaps allowed (real-time data)
    
    max_correlation=0.90,  # Multiple sensors might measure same thing
)

CREDIT_SCORING_RULES = DomainQualityRules(
    domain_name='Credit Scoring',
    description='Rules for credit/financial risk scoring (regulatory + risk management)',
    
    max_missing_pct=0.5,  # Strict (regulatory: FCRA compliance)
    critical_missing_columns=['applicant_id', 'credit_history', 'income', 'debt'],
    
    max_duplicates_pct=0.1,
    allow_exact_duplicates=False,
    
    outlier_detection_method='zscore',
    outlier_threshold_zscore=3.5,
    flag_but_dont_remove=True,  # Manual review (could be fraud or rare legitimate case)
    
    max_imbalance_ratio=5.0,
    critical_minority_size=100,
    
    strict_type_checking=True,
    
    regulatory_compliance={
        'require_applicant_consent': True,
        'max_data_retention_days': 2555,
        'require_adverse_action_notice': True,
        'require_explainability': True,
    }
)


# ============================================================================
# Domain Quality Rules Manager
# ============================================================================

class DomainQualityRulesManager:
    """Manage and apply domain-specific quality rules."""
    
    def __init__(self):
        """Initialize with pre-defined domains."""
        self.domains = {
            'finance': FINANCE_RULES,
            'healthcare': HEALTHCARE_RULES,
            'ecommerce': ECOMMERCE_RULES,
            'sentiment': SENTIMENT_ANALYSIS_RULES,
            'iot': IOT_SENSOR_RULES,
            'credit_scoring': CREDIT_SCORING_RULES,
        }
    
    def get_rules(self, domain: str) -> DomainQualityRules:
        """Get rules for a domain."""
        domain_lower = domain.lower()
        if domain_lower not in self.domains:
            available = ', '.join(self.domains.keys())
            raise ValueError(f"Unknown domain: {domain}. Available: {available}")
        return self.domains[domain_lower]
    
    def list_domains(self) -> Dict[str, str]:
        """List all available domains with descriptions."""
        return {
            name: rules.description 
            for name, rules in self.domains.items()
        }
    
    def create_custom_rules(self, domain_name: str, **kwargs) -> DomainQualityRules:
        """Create custom rules for a new domain."""
        rules = DomainQualityRules(
            domain_name=domain_name,
            description=kwargs.pop('description', f'Custom rules for {domain_name}'),
            **kwargs
        )
        return rules
    
    def register_domain(self, domain_name: str, rules: DomainQualityRules):
        """Register a custom domain."""
        self.domains[domain_name.lower()] = rules
        print(f"✓ Registered domain: {domain_name}")
    
    def apply_rules(self, quality_report: Dict, rules: DomainQualityRules) -> Dict:
        """
        Apply domain rules to quality report and determine if data is acceptable.
        
        Args:
            quality_report: Quality detection results
            rules: Domain-specific rules
            
        Returns:
            dict with:
              - 'passes_checks': bool
              - 'violations': list of rule violations
              - 'warnings': list of warnings
              - 'severity_level': 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
        """
        violations = []
        warnings = []
        
        # Check missing values
        missing = quality_report.get('missing_values', 0)
        if missing > rules.max_missing_pct:
            violations.append(
                f"Missing values ({missing:.2f}%) exceeds domain limit ({rules.max_missing_pct}%)"
            )
        
        # Check duplicates
        duplicates = quality_report.get('duplicates', 0)
        if duplicates > rules.max_duplicates_pct:
            violations.append(
                f"Duplicates ({duplicates:.2f}%) exceeds domain limit ({rules.max_duplicates_pct}%)"
            )
        
        # Check outliers
        outliers = quality_report.get('outliers', 0)
        if rules.flag_but_dont_remove and outliers > 10:
            warnings.append(
                f"Domain requires manual review of {outliers}% outliers (domain rule: flag_but_dont_remove)"
            )
        
        # Check imbalance
        imbalance = quality_report.get('imbalance_ratio', 1.0)
        if imbalance > rules.max_imbalance_ratio:
            violations.append(
                f"Class imbalance ratio ({imbalance:.1f}:1) exceeds domain limit ({rules.max_imbalance_ratio:.1f}:1)"
            )
        
        # Check correlation
        correlation = quality_report.get('high_correlation_pct', 0)
        if correlation > 20 and not rules.allow_multicollinearity:
            warnings.append(
                f"High feature correlation ({correlation:.1f}%) detected. Domain rule: no multicollinearity allowed"
            )
        
        # Determine severity
        if violations:
            if any('Missing values' in v or 'Duplicates' in v for v in violations):
                severity = 'CRITICAL'
            elif any('imbalance' in v.lower() for v in violations):
                severity = 'HIGH'
            else:
                severity = 'MEDIUM'
        elif warnings:
            severity = 'LOW'
        else:
            severity = 'PASS'
        
        return {
            'domain': rules.domain_name,
            'passes_checks': len(violations) == 0,
            'violations': violations,
            'warnings': warnings,
            'severity_level': severity,
            'rules_applied': {
                'max_missing_pct': rules.max_missing_pct,
                'max_duplicates_pct': rules.max_duplicates_pct,
                'max_imbalance_ratio': rules.max_imbalance_ratio,
                'outlier_method': rules.outlier_detection_method,
            }
        }
    
    def save_rules(self, rules: DomainQualityRules, filepath: str):
        """Save rules to JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(rules.to_dict(), f, indent=2)
        
        print(f"✓ Rules saved to {filepath}")
    
    def load_rules(self, filepath: str) -> DomainQualityRules:
        """Load rules from JSON."""
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        
        rules = DomainQualityRules(**data)
        print(f"✓ Rules loaded from {filepath}")
        return rules


# ============================================================================
# Demo
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("DOMAIN-SPECIFIC QUALITY RULES - DEMO")
    print("="*70)
    
    manager = DomainQualityRulesManager()
    
    # Show available domains
    print("\n[1/3] Available Domains:")
    for domain, description in manager.list_domains().items():
        print(f"  • {domain:20} → {description}")
    
    # Get rules for a domain
    print("\n[2/3] Finance Domain Rules:")
    finance_rules = manager.get_rules('finance')
    print(f"  Max missing values: {finance_rules.max_missing_pct}%")
    print(f"  Max duplicates: {finance_rules.max_duplicates_pct}%")
    print(f"  Max imbalance: {finance_rules.max_imbalance_ratio}:1")
    print(f"  Regulatory compliance: {list(finance_rules.regulatory_compliance.keys())}")
    
    # Apply rules to example quality report
    print("\n[3/3] Apply Rules to Sample Data:")
    
    quality_report = {
        'missing_values': 0.05,  # 0.05% missing
        'duplicates': 0.3,       # 0.3% duplicates
        'outliers': 2.5,         # 2.5% outliers
        'imbalance_ratio': 5.0,  # 5:1 imbalance
        'high_correlation_pct': 10,
    }
    
    # Test with Finance domain
    print("\n  Test with FINANCE domain:")
    result = manager.apply_rules(quality_report, finance_rules)
    print(f"    Passes checks: {result['passes_checks']}")
    print(f"    Severity: {result['severity_level']}")
    if result['violations']:
        for v in result['violations']:
            print(f"    ✗ {v}")
    if result['warnings']:
        for w in result['warnings']:
            print(f"    ⚠ {w}")
    
    # Test with E-commerce domain
    print("\n  Test with E-COMMERCE domain:")
    ecom_rules = manager.get_rules('ecommerce')
    result = manager.apply_rules(quality_report, ecom_rules)
    print(f"    Passes checks: {result['passes_checks']}")
    print(f"    Severity: {result['severity_level']}")
    if result['violations']:
        for v in result['violations']:
            print(f"    ✗ {v}")
    if result['warnings']:
        for w in result['warnings']:
            print(f"    ⚠ {w}")
    
    # Test with Healthcare domain
    print("\n  Test with HEALTHCARE domain:")
    health_rules = manager.get_rules('healthcare')
    result = manager.apply_rules(quality_report, health_rules)
    print(f"    Passes checks: {result['passes_checks']}")
    print(f"    Severity: {result['severity_level']}")
    if result['violations']:
        for v in result['violations']:
            print(f"    ✗ {v}")
    if result['warnings']:
        for w in result['warnings']:
            print(f"    ⚠ {w}")
    
    print("\n" + "="*70)
    print("✓ DEMO COMPLETE")
    print("="*70)
    print("\nKey Insight:")
    print("  Same data (missing: 0.05%, duplicates: 0.3%, imbalance: 5:1)")
    print("  → PASSES finance rules (strict)")
    print("  → PASSES e-commerce rules (loose)")
    print("  → Depends on healthcare rules")
    print("\nDifferent domains, different standards!\n")
