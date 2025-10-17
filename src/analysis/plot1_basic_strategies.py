"""
Plot 1: Basic Narrative Strategies Across All Languages
Shows Vanilla, Narrative Intersection, and Narrative Union for BG, EN, HI, PT, RU
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

# Read the data
df = pd.read_csv('results/openai_gpt-5-nano/summary.csv')

# Filter for vanilla, narrative intersection, and narrative union
# For English, we'll use the 2-agent version as the default "Narrative Intersection"
df_filtered = df.copy()

# Standardize English configs for plot 1
df_filtered.loc[(df_filtered['Language'] == 'EN') & (df_filtered['Config'] == 'Narrative Intersection (2 agents)'), 'Config'] = 'Narrative Intersection'

configs_plot1 = ['Vanilla', 'Narrative Intersection', 'Narrative Union']
df_plot1 = df_filtered[df_filtered['Config'].isin(configs_plot1)].copy()

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))

# Create grouped bar plot
languages = ['BG', 'EN', 'HI', 'PT', 'RU']
x = np.arange(len(languages))
width = 0.25

# Define colors for each strategy
colors = {
    'Vanilla': '#66c2a5',
    'Narrative Intersection': '#8da0cb',
    'Narrative Union': '#fc8d62'
}

# Plot F1 samples
for i, config in enumerate(configs_plot1):
    values = []
    for lang in languages:
        val = df_plot1[(df_plot1['Language'] == lang) & (df_plot1['Config'] == config)]['f1 samples']
        values.append(val.values[0] if len(val) > 0 else 0)
    
    ax.bar(x + i*width, values, width, label=config, color=colors[config], alpha=0.85, edgecolor='white', linewidth=1.5)

ax.set_xlabel('Language', fontsize=13, fontweight='bold')
ax.set_ylabel('F1 Samples', fontsize=13, fontweight='bold')
ax.set_title('Basic Narrative Strategies Across Languages', fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x + width)
ax.set_xticklabels(languages, fontsize=12)
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), framealpha=0.95, fontsize=11, edgecolor='black')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 0.7)
ax.set_axisbelow(True)

# Add value labels on bars
for i, config in enumerate(configs_plot1):
    values = []
    for lang in languages:
        val = df_plot1[(df_plot1['Language'] == lang) & (df_plot1['Config'] == config)]['f1 samples']
        values.append(val.values[0] if len(val) > 0 else 0)
    
    for j, v in enumerate(values):
        if v > 0:
            ax.text(x[j] + i*width, v + 0.012, f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('results/openai_gpt-5-nano/plot1_basic_strategies.png', dpi=300, bbox_inches='tight')
print("Plot 1 saved:")
print("  - results/openai_gpt-5-nano/plot1_basic_strategies.png")

# Summary statistics
print("\nBest performing strategy per language:")
for lang in languages:
    lang_data = df_plot1[df_plot1['Language'] == lang]
    best = lang_data.loc[lang_data['f1 samples'].idxmax()]
    print(f"  {lang}: {best['Config']} (F1 Samples = {best['f1 samples']:.3f})")

plt.close()
