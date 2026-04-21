#!/usr/bin/env python3
# /// script
# dependencies = ["matplotlib", "numpy", "pandas"]
# ///
"""Analyze grid search results from GPU batch simulation."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

CSV_PATH = Path(__file__).parent / "grid_search_results.csv"
OUT_DIR = Path(__file__).parent / "charts"
OUT_DIR.mkdir(exist_ok=True)
OBSIDIAN_DIR = Path.home() / "Obsidian" / "Basic Memory" / "Reports" / "cascade" / "attachments"
OBSIDIAN_DIR.mkdir(parents=True, exist_ok=True)

# Read CSV, skip the header lines before the actual CSV header
lines = CSV_PATH.read_text().strip().split('\n')
# Find the CSV header line
header_idx = None
for i, line in enumerate(lines):
    if line.startswith('seed,'):
        header_idx = i
        break

if header_idx is None:
    raise ValueError("Could not find CSV header in grid_search_results.csv")

# Check for summary lines at end
data_lines = []
for line in lines[header_idx+1:]:
    if line.startswith('#') or line.startswith('SUMMARY') or line.strip() == '':
        break
    data_lines.append(line)

# Parse - handle spaces around commas from Mojo's print
header = [h.strip() for h in lines[header_idx].split(',')]
rows = []
for line in data_lines:
    vals = [v.strip() for v in line.split(',')]
    if len(vals) == len(header):
        rows.append(vals)

df = pd.DataFrame(rows, columns=header)
# Convert types
for col in ['seed', 'active', 'support', 'oppose', 'jail', 'revolution']:
    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
for col in ['pp_mean', 'sec_density', 'epsilon', 'threshold']:
    df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)

df = df.dropna()
print(f"Loaded {len(df)} simulations")
print(f"Parameters: {df.nunique().to_dict()}")
print()

# Revolution rate by parameter
rev_rate = df.groupby(df.columns.tolist()[:5].copy())['revolution'].mean()
total_revolutions = df['revolution'].sum()
total_sims = len(df)
print(f"Overall revolution rate: {total_revolutions}/{total_sims} = {total_revolutions/total_sims:.1%}")
print()

# ===== Chart 1: Revolution rate heatmap - pp_mean vs sec_density =====
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Aggregate revolution rate by pp_mean and sec_density (averaged over seeds, eps, threshold)
pivot_rev = df.groupby(['pp_mean', 'sec_density'])['revolution'].mean().reset_index()
pivot_rev['revolution'] = pivot_rev['revolution'].astype(float)
pivot_table = pivot_rev.pivot(index='pp_mean', columns='sec_density', values='revolution').astype(float)

ax = axes[0, 0]
im = ax.imshow(pivot_table.values, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1,
               origin='lower')
ax.set_xticks(range(len(pivot_table.columns)))
ax.set_xticklabels([f'{x:.3f}' for x in pivot_table.columns], rotation=45, fontsize=8)
ax.set_yticks(range(len(pivot_table.index)))
ax.set_yticklabels([f'{x:.2f}' for x in pivot_table.index], fontsize=8)
ax.set_xlabel('Security Density', fontsize=11)
ax.set_ylabel('PP Mean (political preference)', fontsize=11)
ax.set_title('Revolution Rate: PP Mean vs Security Density', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, label='Revolution Probability')

# ===== Chart 2: Revolution rate by epsilon and threshold =====
pivot_et = df.groupby(['epsilon', 'threshold'])['revolution'].mean().reset_index()
pivot_et['revolution'] = pivot_et['revolution'].astype(float)
pivot_et_table = pivot_et.pivot(index='epsilon', columns='threshold', values='revolution').astype(float)

ax = axes[0, 1]
im2 = ax.imshow(pivot_et_table.values, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1,
                origin='lower')
ax.set_xticks(range(len(pivot_et_table.columns)))
ax.set_xticklabels([f'{x:.2f}' for x in pivot_et_table.columns], rotation=45, fontsize=8)
ax.set_yticks(range(len(pivot_et_table.index)))
ax.set_yticklabels([f'{x:.2f}' for x in pivot_et_table.index], fontsize=8)
ax.set_xlabel('Threshold', fontsize=11)
ax.set_ylabel('Epsilon (noise)', fontsize=11)
ax.set_title('Revolution Rate: Epsilon vs Threshold', fontsize=13, fontweight='bold')
plt.colorbar(im2, ax=ax, label='Revolution Probability')

# ===== Chart 3: Security density effect (line plot) =====
ax = axes[1, 0]
for pp in sorted(df['pp_mean'].unique()):
    sub = df[df['pp_mean'] == pp]
    by_sec = sub.groupby('sec_density')['revolution'].mean()
    ax.plot(by_sec.index, by_sec.values, 'o-', label=f'pp={pp:.2f}', markersize=4, linewidth=1.5)
ax.set_xlabel('Security Density', fontsize=11)
ax.set_ylabel('Revolution Probability', fontsize=11)
ax.set_title('Security Density Effect by Political Preference', fontsize=13, fontweight='bold')
ax.legend(fontsize=7, ncol=3, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)

# ===== Chart 4: Distribution of final state counts =====
ax = axes[1, 1]
# Average final state composition across all sims
avg_active = df['active'].mean()
avg_support = df['support'].mean()
avg_oppose = df['oppose'].mean()
avg_jail = df['jail'].mean()
total_agents = avg_active + avg_support + avg_oppose + avg_jail

categories = ['Active', 'Support', 'Oppose', 'Jailed']
values = [avg_active, avg_support, avg_oppose, avg_jail]
colors = ['#e74c3c', '#2ecc71', '#3498db', '#95a5a6']
bars = ax.bar(categories, values, color=colors)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{val:.0f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('Average Count (out of ~1120 citizens)', fontsize=11)
ax.set_title('Average Final State Distribution', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig.savefig(OUT_DIR / 'grid_search_analysis.png', dpi=150, bbox_inches='tight')
fig.savefig(OBSIDIAN_DIR / 'cascade-grid-search-analysis.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'grid_search_analysis.png'}")
print(f"Saved: {OBSIDIAN_DIR / 'cascade-grid-search-analysis.png'}")

# ===== Chart 5: Phase diagram - critical transitions =====
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

# Phase diagram: revolution probability as function of pp_mean for different sec_densities
ax = axes2[0]
sec_vals = sorted(df['sec_density'].unique())
colors_sec = plt.cm.viridis(np.linspace(0, 1, len(sec_vals)))
for sd, color in zip(sec_vals, colors_sec):
    sub = df[df['sec_density'] == sd]
    by_pp = sub.groupby('pp_mean')['revolution'].mean()
    ax.plot(by_pp.index, by_pp.values, 'o-', color=color, label=f'{sd:.3f}', markersize=4, linewidth=1.5)
ax.set_xlabel('PP Mean', fontsize=11)
ax.set_ylabel('Revolution Probability', fontsize=11)
ax.set_title('Phase Transition: PP Mean (by Security Density)', fontsize=13, fontweight='bold')
ax.legend(title='Sec Density', fontsize=7, title_fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# Threshold sensitivity
ax = axes2[1]
for eps in sorted(df['epsilon'].unique()):
    sub = df[df['epsilon'] == eps]
    by_th = sub.groupby('threshold')['revolution'].mean()
    ax.plot(by_th.index, by_th.values, 'o-', label=f'eps={eps:.2f}', markersize=4, linewidth=1.5)
ax.set_xlabel('Threshold', fontsize=11)
ax.set_ylabel('Revolution Probability', fontsize=11)
ax.set_title('Threshold Sensitivity (by Epsilon)', fontsize=13, fontweight='bold')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# Variance across seeds (robustness)
ax = axes2[2]
seed_var = df.groupby(['pp_mean', 'sec_density', 'epsilon', 'threshold'])['revolution'].agg(['mean', 'std']).reset_index()
seed_var['std'] = seed_var['std'].fillna(0)
ax.hist(seed_var['std'], bins=30, color='#3498db', edgecolor='white', alpha=0.8)
ax.set_xlabel('Std Dev of Revolution (across 30 seeds)', fontsize=11)
ax.set_ylabel('Count (parameter combos)', fontsize=11)
ax.set_title('Seed Variance Distribution', fontsize=13, fontweight='bold')
ax.axvline(x=seed_var['std'].mean(), color='red', linestyle='--', label=f'Mean: {seed_var["std"].mean():.3f}')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig2.savefig(OUT_DIR / 'grid_search_phase.png', dpi=150, bbox_inches='tight')
fig2.savefig(OBSIDIAN_DIR / 'cascade-grid-search-phase.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'grid_search_phase.png'}")
print(f"Saved: {OBSIDIAN_DIR / 'cascade-grid-search-phase.png'}")

# Print summary statistics
print("\n=== Summary Statistics ===")
print(f"Total simulations: {len(df)}")
print(f"Revolution rate: {df['revolution'].mean():.3f}")
print(f"\nRevolution rate by pp_mean:")
print(df.groupby('pp_mean')['revolution'].mean().to_string())
print(f"\nRevolution rate by sec_density:")
print(df.groupby('sec_density')['revolution'].mean().to_string())
print(f"\nRevolution rate by epsilon:")
print(df.groupby('epsilon')['revolution'].mean().to_string())
print(f"\nRevolution rate by threshold:")
print(df.groupby('threshold')['revolution'].mean().to_string())

# Find critical parameter combinations
print("\n=== Critical Transitions ===")
# Where does revolution probability cross 50%?
by_pp_sec = df.groupby(['pp_mean', 'sec_density'])['revolution'].mean().reset_index()
transitions = by_pp_sec[(by_pp_sec['revolution'] > 0.3) & (by_pp_sec['revolution'] < 0.7)]
if len(transitions) > 0:
    print(f"Parameter combos near 50% revolution ({len(transitions)} found):")
    print(transitions.sort_values('revolution').to_string(index=False))
else:
    print("No parameter combos near 50% revolution threshold found")
