# Data Quality Detection Framework (DQDF)

An automated framework for detecting data quality issues and measuring their impact on machine learning model performance.

## 🎯 Overview

This framework:
- **Detects** 7 key data quality dimensions (missing values, duplicates, outliers, imbalance, etc.)
- **Measures** impact of each issue on ML model performance (F1 score)
- **Prioritizes** fixes based on predicted performance improvement
- **Fixes** data automatically with configurable strategies
- **Reports** quality score (0-100) and actionable recommendations

## 📊 Quick Start

### 1. Clone & Install

```bash
# Clone repository
git clone <your-repo-url>
cd data-quality-framework

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Full Experiments (One Command)

```bash
python experiments/run_experiments.py
```

**What it does:**
- Downloads 3 public datasets (Adult, Titanic, Credit Card Fraud)
- Detects quality issues on each
- Trains ML models (Random Forest, Logistic Regression, XGBoost) on original data
- Applies quality fixes
- Retrains models and measures F1 improvement
- Saves results to `results/results.json`
- Generates summary report to `results/summary_report.txt`

**Estimated Runtime:** 15-30 minutes (depending on internet speed and CPU)

### 3. Check Results

```bash
cat results/summary_report.txt
```

Results show:
- Quality score for each dataset
- F1 improvement from fixes
- Which models benefited most

## 📁 Project Structure

```
data-quality-framework/
│
├── src/
│   ├── quality_detector.py       # Detect quality issues
│   ├── quality_fixes.py          # Apply fixes
│   └── impact_analyzer.py        # Measure ML performance impact
│
├── experiments/
│   └── run_experiments.py        # Main experiment runner
│
├── paper/
│   └── paper_template.md         # IEEE paper template (fill in your results)
│
├── results/
│   ├── results.json              # Metrics (generated after running)
│   └── summary_report.txt        # Summary (generated after running)
│
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── setup.py                      # Install as package (optional)
```

## 🔧 Detailed Usage

### Option A: Run Full Pipeline (Recommended)

```bash
python experiments/run_experiments.py
```

### Option B: Custom Dataset

```python
import pandas as pd
import sys
sys.path.insert(0, 'src')

from quality_detector import QualityDetector
from quality_fixes import QualityFixer
from impact_analyzer import ImpactAnalyzer

# Load your dataset
df = pd.read_csv('your_data.csv')

# 1. Detect quality issues
detector = QualityDetector()
results = detector.detect_all(df)
detector.print_report(df, results)

# 2. Fix issues
fixer = QualityFixer()
config = {
    'missing_values': 'drop',
    'duplicates': True,
    'outliers': True,
    'imbalance': False,
    'scale': True,
}
df_fixed = fixer.apply_fixes(df, config)

# 3. Analyze impact on ML
analyzer = ImpactAnalyzer()
impact = analyzer.analyze_impact(df, df_fixed, target_col='target', models=['rf', 'lr'])
analyzer.print_impact_report(impact, df, df_fixed)
```

### Option C: Individual Components

#### Detect Quality Issues Only

```python
from src.quality_detector import QualityDetector

detector = QualityDetector()
results = detector.detect_all(df)

# Quality score (0-100)
quality_score = detector.generate_quality_score(results)
print(f"Quality Score: {quality_score:.1f}/100")

# Print detailed report
detector.print_report(df, results)
```

#### Apply Fixes

```python
from src.quality_fixes import QualityFixer

fixer = QualityFixer()

# Apply specific fixes
df_fixed = fixer.fix_missing_values(df, strategy='drop')
df_fixed = fixer.remove_duplicates(df_fixed)
df_fixed = fixer.remove_outliers(df_fixed, method='iqr')

# Or apply all at once with config
config = {
    'missing_values': 'drop',
    'duplicates': True,
    'outliers': True,
}
df_fixed = fixer.apply_fixes(df, config)
```

#### Measure ML Impact

```python
from src.impact_analyzer import ImpactAnalyzer

analyzer = ImpactAnalyzer()

# Compare original vs fixed
impact = analyzer.analyze_impact(df, df_fixed, target_col='income', models=['rf', 'xgb'])

# For each model:
# - Original F1 score
# - Fixed F1 score
# - F1 improvement
```

## 📈 Output Examples

### Quality Detection Report

```
============================================================
DATA QUALITY REPORT
============================================================
Dataset shape: (32561, 14)
Overall Quality Score: 72.3/100
============================================================

Missing Values:
  Severity: medium
  overall_missing_pct: 4.2

Duplicates:
  Severity: low
  percentage: 0.23

Outliers (IQR):
  Severity: high
  total_outlier_records: 2341

Class Imbalance:
  Severity: medium
  imbalance_ratio: 3.1
```

### Impact Analysis Report

```
======================================================================
IMPACT ANALYSIS REPORT
======================================================================
Original data shape: (32561, 14)
Fixed data shape: (30154, 14)
Rows removed/changed: 2407
======================================================================

RANDOM FOREST Model:
  Original F1:   0.6234
  Fixed F1:      0.6891
  F1 Improvement: +0.0657 (+10.53%)

  Original Accuracy: 0.8450
  Fixed Accuracy:    0.8612
  Accuracy Improvement: +0.0162
```

## 🧪 Datasets Included

### 1. Adult (UCI)
- **Task:** Predict income (binary: ≤50K vs >50K)
- **Size:** 32,561 rows, 14 features
- **Issues:** 4.2% missing, 3.1:1 imbalance
- **Download:** Automatic

### 2. Titanic (Kaggle)
- **Task:** Predict survival
- **Size:** 891 rows, 12 features
- **Issues:** 25% missing in Age, 77% in Cabin
- **Download:** Automatic

### 3. Credit Card Fraud (Kaggle)
- **Task:** Detect fraudulent transactions
- **Size:** 284,807 rows, 30 features (first 10K used)
- **Issues:** 500:1 class imbalance, scaled features with outliers
- **Download:** Automatic (subsampled for speed)

## 🔧 Customization

### Change Quality Thresholds

Edit `src/quality_detector.py`:

```python
def _check_completeness(self, df: pd.DataFrame) -> Dict:
    missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()
    severity = 'critical' if any(v > 30 for v in missing_pct.values()) else 'medium'
    # ↑ Change 30 to your threshold
```

### Change Fix Strategy

Edit `experiments/run_experiments.py`:

```python
fix_config = {
    'missing_values': 'median',  # Change from 'drop' to 'median'
    'duplicates': True,
    'outliers': True,
    'imbalance': True,           # Add imbalance fixing
    'scale': True,
}
```

### Add New Datasets

```python
def load_my_dataset(self):
    df = pd.read_csv('my_data.csv')
    return df, 'target_column_name'

# Add to datasets list:
datasets = [
    ('My Dataset', self.load_my_dataset),
    # ...
]
```

## 📝 Writing Your Paper

1. **Copy template:**
   ```bash
   cp paper/paper_template.md paper/my_paper.md
   ```

2. **Run experiments to generate results:**
   ```bash
   python experiments/run_experiments.py
   ```

3. **Fill in results** from `results/results.json` into paper sections 5.1-5.4

4. **Write interpretation** in section 6 (Discussion)

5. **Convert to PDF:**
   ```bash
   pandoc paper/my_paper.md -o paper/my_paper.pdf
   ```

## 🧪 Testing

```bash
# Run quick test on smaller dataset
python -m pytest tests/
```

## 📊 Interpreting Results

### Quality Score Interpretation

| Score | Meaning | Action |
|-------|---------|--------|
| 90-100 | Excellent | Can use as-is |
| 70-90 | Good | Minor fixes recommended |
| 50-70 | Fair | Significant cleaning needed |
| <50 | Poor | Major data quality issues |

### F1 Improvement Interpretation

| Improvement | Meaning |
|-------------|---------|
| >15% | Significant gain from fixes |
| 5-15% | Moderate improvement |
| 1-5% | Minor improvement |
| <1% | Negligible effect |

## ⚡ Performance Tips

### Speed Up Experiments

1. **Use fewer cross-validation folds:**
   ```python
   # In impact_analyzer.py, change cv=5 to cv=3
   ```

2. **Use smaller datasets:**
   ```python
   df = df.sample(n=5000, random_state=42)
   ```

3. **Disable XGBoost** (slowest model):
   ```python
   models=['rf', 'lr']  # Skip XGBoost
   ```

### Reduce Memory Usage

```python
df = df.sample(frac=0.5, random_state=42)  # Use 50% of data
```

## 🐛 Troubleshooting

### Error: "Module not found"
```bash
pip install -r requirements.txt
```

### Error: "Memory error"
- Reduce dataset size
- Use fewer cross-validation folds
- Skip slow models (XGBoost)

### Error: "Internet connection"
- Download datasets manually and load locally
- Comment out `.load_*_dataset()` calls and load from disk

## 📚 References

See `paper/paper_template.md` for full references list (15+ papers cited)

Key references:
- Batini et al. (2009) - Data quality definitions
- Rahm & Do (2000) - Data quality problems
- [Your additional research findings]

## 🤝 Contributing

Ways to extend this framework:

1. **Add more datasets** - Healthcare, finance, social media
2. **Add quality detectors** - Temporal consistency, text quality
3. **Add fix strategies** - KNN imputation, SMOTE oversampling
4. **Add models** - Neural networks, ensemble methods
5. **Meta-learning** - Predict which fixes will help *before* applying them

## 📜 License

MIT License - Feel free to use for research and teaching

## ✉️ Contact

For questions or issues:
- Create an issue on GitHub
- Email: [your-email@college.edu]

---

## ⚡ Next Steps

1. ✅ Install requirements
2. ✅ Run full experiments (`python experiments/run_experiments.py`)
3. ✅ Check results (`cat results/summary_report.txt`)
4. ✅ Fill paper template with your results
5. ✅ Submit to IEEE/conference!

**Good luck with your research! 🔥**
