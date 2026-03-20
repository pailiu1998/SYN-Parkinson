#!/usr/bin/env python
"""
Visualize synthetic validation experiment results
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Results data
results = {
    'Experiment': [
        'Smile+Finger\n(2 real)',
        'Smile+Speech\n(2 real)', 
        'Finger+Speech\n(2 real)',
        'All 3 modalities\n(gold standard)',
        'Smile+Finger+\nSynth Speech',
        'Smile+Synth Finger\n+Speech',
        'Synth Smile+\nFinger+Speech'
    ],
    'AUROC': [0.810, 0.939, 0.921, 0.831, 0.514, 0.831, 0.615],
    'Type': ['2-real', '2-real', '2-real', 'gold', '2R+1S', '2R+1S', '2R+1S'],
    'Color': ['#3498db', '#3498db', '#3498db', '#2ecc71', '#e74c3c', '#2ecc71', '#e74c3c']
}

df = pd.DataFrame(results)

# Create bar plot
fig, ax = plt.subplots(figsize=(14, 8))

bars = ax.bar(range(len(df)), df['AUROC'], color=df['Color'], alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, df['AUROC'])):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{val:.3f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Styling
ax.set_xlabel('Experiment', fontsize=14, fontweight='bold')
ax.set_ylabel('AUROC', fontsize=14, fontweight='bold')
ax.set_title('Synthetic Data Validation Results\nComparing Real vs Synthetic Modalities', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df['Experiment'], rotation=15, ha='right', fontsize=11)
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=0.831, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Gold Standard (0.831)')
ax.axhline(y=0.5, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Random (0.5)')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498db', label='2 Real Modalities (Baseline)'),
    Patch(facecolor='#2ecc71', label='Gold Standard / Good Synthetic'),
    Patch(facecolor='#e74c3c', label='Bad Synthetic Data')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

plt.tight_layout()
plt.savefig('/localdisk2/pliu/park_multitask_fusion-main/results/synthetic_validation/results_comparison.png', 
            dpi=300, bbox_inches='tight')
print("✅ Plot saved to results_comparison.png")

# Print summary table
print("\n" + "="*80)
print("RESULTS SUMMARY TABLE")
print("="*80)
print(df[['Experiment', 'AUROC', 'Type']].to_string(index=False))
print("="*80)
print("\n🎯 KEY FINDINGS:")
print("  • Best 2-modality: Speech + Smile (0.939)")
print("  • Synthetic Finger works perfectly: 0.831 (= gold standard)")
print("  • Synthetic Speech/Smile fail: < 0.65")
print("\n")


