"""
Visualization Script - Generate publication-quality charts
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def get_results_dir():
    """Get results directory"""
    return os.path.join(os.path.dirname(__file__), '..', 'results')

def load_results():
    """Load results from JSON"""
    results_dir = get_results_dir()
    results_file = os.path.join(results_dir, 'results.json')
    print(f"Loading from: {results_file}")
    with open(results_file, 'r') as f:
        return json.load(f)

def plot_quality_scores(results):
    """Plot quality scores"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    datasets = list(results.keys())
    scores = [results[d]['quality_score'] for d in datasets]
    colors = ['#2ecc71' if s > 75 else '#f39c12' for s in scores]
    
    bars = ax.bar(datasets, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Quality Score', fontsize=12, fontweight='bold')
    ax.set_title('Data Quality Scores Across Datasets', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    results_dir = get_results_dir()
    save_path = os.path.join(results_dir, 'fig_quality_scores.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  OK fig_quality_scores.png")
    plt.close()

def plot_f1_improvement(results):
    """Plot F1 improvements"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    datasets = list(results.keys())
    models = list(results[datasets[0]]['impact_analysis'].keys())
    
    x = np.arange(len(datasets))
    width = 0.35
    
    for i, model in enumerate(models):
        values = [results[d]['impact_analysis'][model]['improvement']['f1'] for d in datasets]
        ax.bar(x + i*width, values, width, label=model.upper(), alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('F1 Score Improvement', fontsize=12, fontweight='bold')
    ax.set_title('F1 Score Improvement After Data Quality Fixes', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    results_dir = get_results_dir()
    save_path = os.path.join(results_dir, 'fig_f1_improvement.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  OK fig_f1_improvement.png")
    plt.close()

def plot_data_reduction(results):
    """Plot data reduction"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    datasets = list(results.keys())
    data_removed = []
    data_retained = []
    
    for dataset in datasets:
        orig_rows = results[dataset]['data_shape_original'][0]
        fixed_rows = results[dataset]['data_shape_fixed'][0]
        removed = orig_rows - fixed_rows
        data_removed.append(removed)
        data_retained.append(fixed_rows)
    
    x = np.arange(len(datasets))
    width = 0.6
    
    ax.bar(x, data_retained, width, label='Rows Retained', alpha=0.8, color='#3498db', edgecolor='black')
    ax.bar(x, data_removed, width, bottom=data_retained, label='Rows Removed', alpha=0.8, color='#e74c3c', edgecolor='black')
    
    for i, (removed, retained) in enumerate(zip(data_removed, data_retained)):
        total = removed + retained
        pct = (removed / total * 100) if total > 0 else 0
        ax.text(i, total + 100, f'{pct:.1f}%\nremoved', ha='center', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Number of Rows', fontsize=12, fontweight='bold')
    ax.set_title('Data Reduction: Rows Removed by Quality Fixes', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    results_dir = get_results_dir()
    save_path = os.path.join(results_dir, 'fig_data_reduction.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  OK fig_data_reduction.png")
    plt.close()

def plot_performance_comparison(results):
    """Plot performance comparison"""
    fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
    
    if len(results) == 1:
        axes = [axes]
    
    datasets = list(results.keys())
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        impact = results[dataset]['impact_analysis']
        
        models = list(impact.keys())
        original_f1 = [impact[m]['original']['f1'] for m in models]
        fixed_f1 = [impact[m]['fixed']['f1'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax.bar(x - width/2, original_f1, width, label='Original', alpha=0.8, color='#e74c3c', edgecolor='black')
        ax.bar(x + width/2, fixed_f1, width, label='After Fix', alpha=0.8, color='#2ecc71', edgecolor='black')
        
        ax.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{dataset}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in models])
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    fig.suptitle('Model Performance: Original vs After Quality Fixes', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    results_dir = get_results_dir()
    save_path = os.path.join(results_dir, 'fig_performance_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  OK fig_performance_comparison.png")
    plt.close()

def main():
    """Generate visualizations"""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60 + "\n")
    
    results_dir = get_results_dir()
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        results = load_results()
        
        print("Generating charts...")
        plot_quality_scores(results)
        plot_f1_improvement(results)
        plot_data_reduction(results)
        plot_performance_comparison(results)
        
        print("\n" + "="*60)
        print("SUCCESS! All visualizations generated!")
        print("="*60)
        print(f"\nLocation: {results_dir}")
        print("\nCharts created:")
        for f in ['fig_quality_scores.png', 'fig_f1_improvement.png', 'fig_data_reduction.png', 'fig_performance_comparison.png']:
            print(f"  - {f}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()