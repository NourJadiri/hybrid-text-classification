"""
Plot 2: Advanced Multi-Agent Strategies
Shows Full Intersection for EN & HI, Full Union & Full Mixed for RU (vs Vanilla baselines)
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

# Prepare data for plot 2
plot2_data = []

# English Full Intersection vs Vanilla
plot2_data.append({
    'Language': 'EN',
    'Strategy': 'Vanilla',
    'f1_samples': df[(df['Language'] == 'EN') & (df['Config'] == 'Vanilla')]['f1 samples'].values[0]
})
plot2_data.append({
    'Language': 'EN',
    'Strategy': 'Full Intersection',
    'f1_samples': df[(df['Language'] == 'EN') & (df['Config'] == 'Full Intersection')]['f1 samples'].values[0]
})

# Hindi Full Intersection vs Vanilla
plot2_data.append({
    'Language': 'HI',
    'Strategy': 'Vanilla',
    'f1_samples': df[(df['Language'] == 'HI') & (df['Config'] == 'Vanilla')]['f1 samples'].values[0]
})
plot2_data.append({
    'Language': 'HI',
    'Strategy': 'Full Intersection',
    'f1_samples': df[(df['Language'] == 'HI') & (df['Config'] == 'Full Intersection')]['f1 samples'].values[0]
})

# Russian Full Union, Full Mixed vs Vanilla
plot2_data.append({
    'Language': 'RU',
    'Strategy': 'Vanilla',
    'f1_samples': df[(df['Language'] == 'RU') & (df['Config'] == 'Vanilla')]['f1 samples'].values[0]
})
plot2_data.append({
    'Language': 'RU',
    'Strategy': 'Full Union',
    'f1_samples': df[(df['Language'] == 'RU') & (df['Config'] == 'Full Union')]['f1 samples'].values[0]
})
plot2_data.append({
    'Language': 'RU',
    'Strategy': 'Full Mixed',
    'f1_samples': df[(df['Language'] == 'RU') & (df['Config'] == 'Full Mixed')]['f1 samples'].values[0]
})

df_plot2 = pd.DataFrame(plot2_data)

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))

# Define colors for plot 2
colors_plot2 = {
    'Vanilla': '#66c2a5',
    'Full Intersection': '#8da0cb',
    'Full Union': '#fc8d62',
    'Full Mixed': '#e78ac3'
}

# Group by language
languages_plot2 = ['EN', 'HI', 'RU']
x_plot2 = np.arange(len(languages_plot2))

# Track which labels we've added to legend
added_labels = set()

# For each language, plot its strategies
for lang_idx, lang in enumerate(languages_plot2):
    lang_data = df_plot2[df_plot2['Language'] == lang]
    strategies = lang_data['Strategy'].unique()
    
    # Calculate positions for bars
    if lang == 'RU':
        width_p2 = 0.2
        positions = [lang_idx - 0.3, lang_idx - 0.1, lang_idx + 0.1]
    else:
        width_p2 = 0.25
        positions = [lang_idx - 0.15, lang_idx + 0.15]
    
    for i, strategy in enumerate(strategies):
        value = lang_data[lang_data['Strategy'] == strategy]['f1_samples'].values[0]
        
        # Only add label if not already added
        label = strategy if strategy not in added_labels else ''
        if label:
            added_labels.add(strategy)
        
        ax.bar(positions[i], value, width_p2, label=label, 
               color=colors_plot2[strategy], alpha=0.85, edgecolor='white', linewidth=1.5)
        ax.text(positions[i], value + 0.012, f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xlabel('Language', fontsize=13, fontweight='bold')
ax.set_ylabel('F1 Samples', fontsize=13, fontweight='bold')
ax.set_title('Advanced Multi-Agent Strategies', fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x_plot2)
ax.set_xticklabels(languages_plot2, fontsize=12)

# Create custom legend with all strategies in order
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors_plot2['Vanilla'], label='Vanilla', alpha=0.85, edgecolor='white', linewidth=1.5),
    Patch(facecolor=colors_plot2['Full Intersection'], label='Full Intersection', alpha=0.85, edgecolor='white', linewidth=1.5),
    Patch(facecolor=colors_plot2['Full Union'], label='Full Union', alpha=0.85, edgecolor='white', linewidth=1.5),
    Patch(facecolor=colors_plot2['Full Mixed'], label='Full Mixed', alpha=0.85, edgecolor='white', linewidth=1.5)
]
ax.legend(handles=legend_elements, loc='upper left', framealpha=0.95, fontsize=11, edgecolor='black')

ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 0.75)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('results/openai_gpt-5-nano/plot2_advanced_strategies.png', dpi=300, bbox_inches='tight')
print("Plot 2 saved:")
print("  - results/openai_gpt-5-nano/plot2_advanced_strategies.png")

# Summary statistics
print("\nImprovement from Vanilla:")
print(f"  EN Full Intersection: {df_plot2[(df_plot2['Language']=='EN') & (df_plot2['Strategy']=='Full Intersection')]['f1_samples'].values[0] - df_plot2[(df_plot2['Language']=='EN') & (df_plot2['Strategy']=='Vanilla')]['f1_samples'].values[0]:+.3f}")
print(f"  HI Full Intersection: {df_plot2[(df_plot2['Language']=='HI') & (df_plot2['Strategy']=='Full Intersection')]['f1_samples'].values[0] - df_plot2[(df_plot2['Language']=='HI') & (df_plot2['Strategy']=='Vanilla')]['f1_samples'].values[0]:+.3f}")
print(f"  RU Full Union: {df_plot2[(df_plot2['Language']=='RU') & (df_plot2['Strategy']=='Full Union')]['f1_samples'].values[0] - df_plot2[(df_plot2['Language']=='RU') & (df_plot2['Strategy']=='Vanilla')]['f1_samples'].values[0]:+.3f}")
print(f"  RU Full Mixed: {df_plot2[(df_plot2['Language']=='RU') & (df_plot2['Strategy']=='Full Mixed')]['f1_samples'].values[0] - df_plot2[(df_plot2['Language']=='RU') & (df_plot2['Strategy']=='Vanilla')]['f1_samples'].values[0]:+.3f}")

plt.close()
