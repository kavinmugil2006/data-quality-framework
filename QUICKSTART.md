# 🚀 QUICK START GUIDE

## What You Have

✅ **Complete working project** with:
- Data quality detection code (quality_detector.py)
- Data fixing code (quality_fixes.py)
- ML impact analysis code (impact_analyzer.py)
- Experiment runner ready to execute (run_experiments.py)
- IEEE paper template with placeholders for results (paper_template.md)

✅ **Zero coding required from you** - just run one command

---

## What You Need to Do

### Step 1: Install (5 minutes)

```bash
# Go to project directory
cd data-quality-framework

# Install all dependencies
pip install -r requirements.txt
```

**Issues?**
- If `pip` not found, use `pip3` instead
- If permission error, add `--user`: `pip install --user -r requirements.txt`

### Step 2: Run Experiments (15-30 minutes)

```bash
python experiments/run_experiments.py
```

**What this does:**
- Automatically downloads 3 datasets (Adult, Titanic, Credit Fraud)
- Detects quality issues on each
- Trains 3 ML models (Random Forest, Logistic Regression, XGBoost) on ORIGINAL data
- Applies quality fixes
- Retrains models on FIXED data
- Measures F1 improvement
- Saves results to `results/results.json`

**You can close your laptop and let it run.**

### Step 3: Check Results (2 minutes)

```bash
cat results/summary_report.txt
```

You'll see something like:

```
========================================
SUMMARY REPORT
========================================

Adult:
  Quality Score: 72.3/100
  Avg F1 Improvement: +0.0657 (10.53%)

Titanic:
  Quality Score: 65.1/100
  Avg F1 Improvement: +0.1234 (19.72%)

Credit Card Fraud:
  Quality Score: 58.4/100
  Avg F1 Improvement: +0.0892 (8.91%)
```

### Step 4: Fill Paper (30 minutes)

```bash
# Open the paper template
open paper/paper_template.md
```

**Find these sections and fill in with your results:**

| Section | What to fill |
|---------|-------------|
| **5.1** Quality Detection Results | Copy from results/results.json |
| **5.2** Impact Analysis Results | Copy F1 improvements from results/results.json |
| **5.3** Issue Impact Ranking | Ranking of which issues matter most |
| **6** Discussion | Your interpretation (2-3 paragraphs) |

**Don't copy blindly.** You must:
- Understand what each result means
- Explain WHY certain datasets improved more
- Discuss which models were most/least robust
- Note limitations

### Step 5: Submit (Depends on venue)

**For IEEE Conference:**
1. Convert to PDF: `pandoc paper/my_paper.md -o my_paper.pdf`
2. Follow conference submission format
3. Submit through conference portal

**For GitHub/Portfolio:**
1. Push to public repo: `git push origin main`
2. Add to your CV/portfolio
3. Link results in README

---

## Files Breakdown

### What Each File Does

| File | Purpose | Do I edit it? |
|------|---------|---|
| `src/quality_detector.py` | Detects quality issues | No (it works) |
| `src/quality_fixes.py` | Fixes the issues | No (it works) |
| `src/impact_analyzer.py` | Trains models & measures impact | No (it works) |
| `experiments/run_experiments.py` | Runs all experiments | No (but can customize) |
| `paper/paper_template.md` | IEEE paper template | **YES - Fill with results** |
| `results/results.json` | Your experiment results | Generated (don't edit) |
| `results/summary_report.txt` | Human-readable summary | Generated (don't edit) |

### What Gets Generated After Running Experiments

```
results/
├── results.json              # Raw numbers (for paper)
└── summary_report.txt        # Pretty summary (to read)
```

---

## Customization (Optional)

### Change Datasets

Edit `experiments/run_experiments.py`, replace the dataset loaders:

```python
# Line ~100, in run_all_experiments()
datasets = [
    ('Adult', self.load_adult_dataset),
    ('Titanic', self.load_titanic_dataset),
    ('Credit Card Fraud', self.load_credit_fraud_dataset),
]

# CHANGE TO:
datasets = [
    ('My Dataset 1', self.load_my_custom_dataset1),
    ('My Dataset 2', self.load_my_custom_dataset2),
]
```

Add your loader function:

```python
def load_my_custom_dataset(self):
    df = pd.read_csv('path/to/my/data.csv')
    return df, 'target_column_name'
```

### Change Models Tested

In `experiments/run_experiments.py`, find:

```python
impact_results = analyzer.analyze_impact(df, df_fixed, target_col, models=['rf', 'lr', 'xgb'])
```

Change to:

```python
impact_results = analyzer.analyze_impact(df, df_fixed, target_col, models=['rf', 'xgb'])  # Skip LR
```

### Change Quality Fixes

In `experiments/run_experiments.py`, find:

```python
fix_config = {
    'missing_values': 'drop',
    'duplicates': True,
    'outliers': True,
    'imbalance': False,
    'scale': True,
}
```

Change strategies:

```python
fix_config = {
    'missing_values': 'median',     # Impute instead of drop
    'duplicates': True,
    'outliers': True,
    'imbalance': True,              # Fix class imbalance
    'scale': True,
}
```

---

## Understanding Your Results

### Quality Score (0-100)

Higher = cleaner data

```
90-100: "Data is very clean, minor issues only"
70-90:  "Data is okay, some fixes recommended"
50-70:  "Data is messy, significant cleaning needed"
<50:    "Data has major quality problems"
```

### F1 Improvement

How much your ML model improves after fixes

```
+20%:  "HUGE improvement from cleaning"
+10%:  "Significant improvement"
+5%:   "Noticeable improvement"
+1%:   "Marginal improvement"
```

### Which Issues Matter Most?

The experiment simulates each issue separately, showing impact:

- **Missing values:** 7% F1 drop ← matters
- **Duplicates:** 0.5% F1 drop ← doesn't matter much
- **Outliers:** 3% F1 drop ← matters somewhat

Use this to prioritize in your discussion.

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'sklearn'"

```bash
pip install scikit-learn
```

### "Pandas not found"

```bash
pip install pandas
```

### "Stuck on downloading data"

The datasets are downloaded from URLs. If internet is slow:

1. Download manually:
   - Adult: https://archive.ics.uci.edu/ml/datasets/Adult
   - Titanic: https://www.kaggle.com/c/titanic/data
   - Credit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

2. Save to local folder

3. Edit `run_experiments.py` to load locally instead of URL

### "MemoryError"

If you run out of RAM:

In `load_credit_fraud_dataset()`:

```python
df = df.sample(n=5000, random_state=42)  # Use 5000 rows instead of 10000
```

### "Takes too long"

Reduce cross-validation folds in `impact_analyzer.py`:

```python
# Change from cv=5 to cv=3
accuracy_cv = cross_val_score(model, X, y, cv=3, scoring='accuracy', n_jobs=-1).mean()
```

---

## Timeline Estimation

| Task | Time | What happens |
|------|------|---|
| Install dependencies | 5 min | Computer downloads packages |
| Run experiments | 20-30 min | Models train (go grab coffee) |
| Check results | 2 min | Read summary_report.txt |
| Fill paper | 30 min | Copy results, write discussion |
| **TOTAL** | **~1-1.5 hours** | **You're done!** |

---

## What NOT to Do

❌ **Don't edit:**
- `quality_detector.py` - It works as-is
- `quality_fixes.py` - It works as-is
- `impact_analyzer.py` - It works as-is

❌ **Don't copy-paste results without understanding them**
- Reviewers will ask "why did this model improve more?"
- You should be able to explain findings

❌ **Don't run experiments and immediately submit**
- Spend 15-20 min understanding what the results mean
- Write a thoughtful discussion section
- That's what makes it research, not just a script

---

## After Paper is Written

### To Submit to IEEE Conference

1. Format as PDF:
   ```bash
   pandoc paper/my_paper.md -o my_paper.pdf
   ```

2. Check requirements:
   - Page limit (usually 6-8 pages)
   - Template (IEEE provides one)
   - Reference format

3. Submit through conference portal

### To Share on GitHub

1. Make repo public
2. Push code + paper + results
3. Add to portfolio/CV

### To Use for Interview Prep

1. Understand every line of code you wrote (know what it does)
2. Be ready to explain:
   - How quality detection works
   - Why fixes improve F1
   - Which datasets benefited most
   - Limitations
3. Bonus: Implement custom fix strategies during interview

---

## Key Numbers to Remember

When someone asks "So what did you build?"

Answer: 

> "An automated framework that detects data quality issues (missing values, outliers, duplicates, imbalance) across **3 public datasets** and measures their impact on **3 ML models** using **controlled experiments**. Results show **X% average F1 improvement** after fixing quality issues. The framework provides an automated quality score (0-100) and prioritizes fixes based on predicted performance impact."

Replace X with your actual average improvement from results.

---

## You're Ready! 🚀

1. ✅ `pip install -r requirements.txt`
2. ✅ `python experiments/run_experiments.py`
3. ✅ Check `results/summary_report.txt`
4. ✅ Fill `paper/paper_template.md`
5. ✅ Submit or share!

**Questions?** Check README.md for detailed documentation.

**Good luck! 🔥**
