# ============================================================================
# READY-TO-PASTE NOTEBOOK CELLS
# Copy these cells directly into your hdh_experiments_final_with_progress.ipynb
# ============================================================================

# CELL 1: Enhanced Imports
# -------------------------
# Add after your existing imports

from pathlib import Path
import sys

# Add path to statistical analysis module
sys.path.append('/home/claude')  # Adjust to where you save the file

from statistical_analysis_enhanced import (
    test_win_rate,
    analyze_cost_reduction,
    stratified_analysis_by_overhead,
    cross_model_comparison,
    plot_cost_comparison_with_ci,
    plot_overhead_scaling_with_stats,
    plot_cross_model_with_significance
)

print("✓ Statistical analysis functions loaded")


# CELL 2: Enhanced Overhead Scaling Plot (replaces your current Figure 7)
# ------------------------------------------------------------------------

# Calculate statistics for overhead experiment
df_overhead = pd.DataFrame(results)  # Your existing results

# Group by overhead and calculate stats
overhead_grouped = df_overhead.groupby('overhead')['cut_cost'].agg([
    ('mean', 'mean'),
    ('std', 'std'),
    ('sem', 'sem'),
    ('count', 'count')
]).reset_index()

# Create enhanced plot with confidence intervals
fig, ax = plt.subplots(figsize=(12, 6))

# Calculate 95% CI
ci_95 = 1.96 * overhead_grouped['sem']

# Plot with shaded CI region
ax.plot(overhead_grouped['overhead'], overhead_grouped['mean'],
       'o-', linewidth=3, markersize=10, 
       color='#2E86AB', label='Mean Cost', zorder=3)

ax.fill_between(overhead_grouped['overhead'],
                overhead_grouped['mean'] - ci_95,
                overhead_grouped['mean'] + ci_95,
                alpha=0.25, color='#2E86AB', 
                label='95% CI', zorder=2)

# Styling
ax.set_xlabel('Overhead Factor', fontsize=14, fontweight='bold')
ax.set_ylabel('Communication Cost (Cut Hyperedges)', fontsize=14, fontweight='bold')
ax.set_title('Partitioning Performance vs Overhead Factor\n(with 95% Confidence Intervals)',
            fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
ax.set_xlim(0.95, max(overhead_grouped['overhead']) + 0.05)

# Add sample size annotation
n_samples = overhead_grouped['count'].iloc[0]
ax.text(0.02, 0.98, f'n = {n_samples} per point', 
       transform=ax.transAxes, fontsize=10,
       verticalalignment='top', bbox=dict(boxstyle='round', 
       facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig7_overhead_with_ci.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✓ Enhanced Figure 7 saved: {OUTPUT_DIR / 'fig7_overhead_with_ci.png'}")
print(f"✓ Mean costs range: {overhead_grouped['mean'].min():.1f} to {overhead_grouped['mean'].max():.1f}")


# CELL 3: Quick Win Rate Analysis (if you have greedy vs METIS data)
# -------------------------------------------------------------------
# NOTE: This assumes you have already run both greedy and METIS
# If not, see INTEGRATION_GUIDE.md for full comparison experiment

# Assuming you have these arrays from your experiments:
# greedy_costs = array of costs from greedy algorithm
# metis_costs = array of costs from METIS algorithm

# Example with your data structure:
# greedy_costs = df_comparison[df_comparison['algorithm']=='greedy']['cut_cost'].values
# metis_costs = df_comparison[df_comparison['algorithm']=='metis']['cut_cost'].values

# If you have the data, uncomment and run:
"""
greedy_wins = np.sum(greedy_costs < metis_costs)
total_workloads = len(greedy_costs)

win_result = test_win_rate(total_workloads, greedy_wins)

print("="*70)
print("WIN RATE ANALYSIS")
print("="*70)
print(f"Total workloads tested: {win_result.total_workloads}")
print(f"Greedy wins: {win_result.greedy_wins} ({win_result.win_rate:.2f}%)")
print(f"METIS wins: {win_result.metis_wins}")
print(f"95% Confidence Interval: [{win_result.ci_lower:.2f}%, {win_result.ci_upper:.2f}%]")
print(f"P-value: {win_result.p_value:.2e}")
print(f"\nFor paper (LaTeX): {win_result.format_latex()}")
print("="*70)
"""


# CELL 4: Cost Reduction Analysis
# --------------------------------
# Again, requires both greedy and METIS data

"""
cost_result = analyze_cost_reduction(greedy_costs, metis_costs)

print("="*70)
print("COST REDUCTION ANALYSIS")
print("="*70)
print(f"Sample size: {cost_result.n_samples}")
print(f"Mean reduction: {cost_result.mean_reduction:.2f}%")
print(f"Median reduction: {cost_result.median_reduction:.2f}%")
print(f"Std deviation: {cost_result.std_reduction:.2f}%")
print(f"95% CI: [{cost_result.ci_lower:.2f}%, {cost_result.ci_upper:.2f}%]")
print(f"Effect size (Cohen's d): {cost_result.effect_size:.2f}")
print(f"P-value (Wilcoxon): {cost_result.p_value:.2e}")
print(f"\nFor paper (LaTeX): {cost_result.format_latex()}")
print("="*70)

# Create violin plot showing distribution
fig, ax = plt.subplots(figsize=(10, 6))

data_to_plot = [greedy_costs, metis_costs]
positions = [1, 2]
labels = ['Greedy\n(capacity-aware)', 'METIS\n(general-purpose)']

parts = ax.violinplot(data_to_plot, positions=positions, 
                      showmeans=True, showmedians=True)

# Color the violins
colors = ['#06A77D', '#A23B72']
for pc, color in zip(parts['bodies'], colors):
    pc.set_facecolor(color)
    pc.set_alpha(0.7)

ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
ax.set_ylabel('Communication Cost (Cut Hyperedges)', fontsize=13, fontweight='bold')
ax.set_title('Cost Distribution: Greedy vs METIS', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add significance annotation
y_max = max(metis_costs.max(), greedy_costs.max())
ax.plot([1, 2], [y_max*1.05, y_max*1.05], 'k-', linewidth=1.5)
p_str = "***" if cost_result.p_value < 0.001 else "**" if cost_result.p_value < 0.01 else "*"
ax.text(1.5, y_max*1.07, p_str, ha='center', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'cost_distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✓ Saved: {OUTPUT_DIR / 'cost_distribution_comparison.png'}")
"""


# CELL 5: Enhanced Cross-Model Plot (Figure 9)
# ---------------------------------------------
# This enhances your existing cross-model comparison

# Assuming df_models exists from your Experiment 3
# Add algorithm labels if running both greedy and METIS per model

fig, ax = plt.subplots(figsize=(14, 8))

colors = {
    'Circuit': '#06A77D',
    'MBQC': '#D4A574', 
    'QW': '#2E86AB',
    'QCA': '#A23B72'
}

models = ['Circuit', 'MBQC', 'QW', 'QCA']

for model in models:
    model_data = df_models[df_models['model'] == model]
    
    # Group by number of qubits and calculate stats
    grouped = model_data.groupby('num_qubits')['cut_cost'].agg([
        'mean', 'std', 'count'
    ])
    
    # Calculate 95% CI
    ci = 1.96 * grouped['std'] / np.sqrt(grouped['count'])
    
    # Plot
    ax.plot(grouped.index, grouped['mean'],
           'o-', label=model, linewidth=2.5, markersize=8,
           color=colors[model], alpha=0.9, zorder=3)
    
    ax.fill_between(grouped.index,
                    grouped['mean'] - ci,
                    grouped['mean'] + ci,
                    alpha=0.15, color=colors[model], zorder=2)

ax.set_xlabel('Number of Qubits', fontsize=14, fontweight='bold')
ax.set_ylabel('Communication Cost (Cut Hyperedges)', fontsize=14, fontweight='bold')
ax.set_title('Cross-Model Partitioning Performance\n(Error bars show 95% CI)',
            fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='upper left', ncol=2, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
ax.set_xscale('log', base=2)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig9_cross_model_enhanced.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✓ Enhanced Figure 9 saved: {OUTPUT_DIR / 'fig9_cross_model_enhanced.png'}")


# CELL 6: Summary Statistics Table for Paper
# -------------------------------------------

print("="*70)
print("SUMMARY STATISTICS FOR PAPER - COPY TO LATEX")
print("="*70)

# Overhead scaling summary
print("\n### Overhead Scaling (Section 6.1)")
print("-" * 50)
overhead_summary = df_overhead.groupby('overhead')['cut_cost'].agg([
    ('mean', lambda x: f"{x.mean():.1f}"),
    ('ci', lambda x: f"±{1.96 * x.std() / np.sqrt(len(x)):.1f}")
])
print(overhead_summary.to_string())

# Cross-model summary
print("\n### Cross-Model Summary (Section 6.2)")
print("-" * 50)
model_summary = df_models.groupby('model')['cut_cost'].agg([
    ('mean', lambda x: f"{x.mean():.1f}"),
    ('median', lambda x: f"{x.median():.1f}"),
    ('std', lambda x: f"{x.std():.1f}")
])
print(model_summary.to_string())

print("\n" + "="*70)
print("✓ Copy these values to your paper!")
print("="*70)


# CELL 7: Export All Results to CSV
# ----------------------------------

# Create comprehensive results CSV
results_export = {
    'overhead_analysis': df_overhead,
    'qpu_scaling': df_qpu if 'df_qpu' in locals() else None,
    'cross_model': df_models
}

for name, df in results_export.items():
    if df is not None:
        filepath = OUTPUT_DIR / f'{name}_results.csv'
        df.to_csv(filepath, index=False)
        print(f"✓ Exported: {filepath}")

print(f"\n✓ All results exported to: {OUTPUT_DIR.absolute()}")


# CELL 8: Generate LaTeX Table Snippets
# --------------------------------------

# Overhead table for paper
overhead_latex = df_overhead.groupby('overhead')['cut_cost'].agg([
    'mean', 'std', 'count'
]).round(2)

overhead_latex['ci_lower'] = (overhead_latex['mean'] - 
                              1.96 * overhead_latex['std'] / 
                              np.sqrt(overhead_latex['count'])).round(2)
overhead_latex['ci_upper'] = (overhead_latex['mean'] + 
                              1.96 * overhead_latex['std'] / 
                              np.sqrt(overhead_latex['count'])).round(2)

latex_table = overhead_latex[['mean', 'ci_lower', 'ci_upper']].to_latex(
    float_format="%.2f",
    column_format='l|ccc',
    caption='Communication costs by overhead factor (95\\% CI)',
    label='tab:overhead_costs'
)

print("="*70)
print("LATEX TABLE FOR PAPER")
print("="*70)
print(latex_table)

# Save to file
with open(OUTPUT_DIR / 'overhead_table.tex', 'w') as f:
    f.write(latex_table)

print(f"\n✓ LaTeX table saved: {OUTPUT_DIR / 'overhead_table.tex'}")
print("="*70)


# ============================================================================
# END OF READY-TO-PASTE CELLS
# ============================================================================

# USAGE NOTES:
# 1. Copy cells above one at a time into your notebook
# 2. Cell 1 goes right after your existing imports
# 3. Cell 2 replaces your current Figure 7 plotting
# 4. Cells 3-4 require you to have both greedy and METIS results
# 5. Cell 5 enhances your existing Figure 9
# 6. Cells 6-8 generate paper-ready outputs
#
# For full details and the greedy vs METIS comparison experiment,
# see INTEGRATION_GUIDE.md