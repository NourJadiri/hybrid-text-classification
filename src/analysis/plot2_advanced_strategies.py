"""
Plot 2: Multi-Agent Strategies Across Languages
Shows Full/Narrative/Subnarrative strategies for EN, Full/Narrative for HI, Full/Narrative/Mixed for RU (vs Vanilla baselines)
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

# English - All strategies
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
# Standardize the English Narrative Intersection name
en_narr_intersection = df[(df['Language'] == 'EN') & (df['Config'].str.contains('Narrative Intersection'))]['f1 samples'].values
plot2_data.append({
    'Language': 'EN',
    'Strategy': 'Narrative Intersection',
    'f1_samples': en_narr_intersection[0] if len(en_narr_intersection) > 0 else 0
})
plot2_data.append({
    'Language': 'EN',
    'Strategy': 'Subnarrative Intersection',
    'f1_samples': df[(df['Language'] == 'EN') & (df['Config'] == 'Subnarrative Intersection')]['f1 samples'].values[0]
})

# Hindi - Full Intersection and Narrative Intersection
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
plot2_data.append({
    'Language': 'HI',
    'Strategy': 'Narrative Intersection',
    'f1_samples': df[(df['Language'] == 'HI') & (df['Config'] == 'Narrative Intersection')]['f1 samples'].values[0]
})

# Russian - All strategies
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
plot2_data.append({
    'Language': 'RU',
    'Strategy': 'Narrative Union',
    'f1_samples': df[(df['Language'] == 'RU') & (df['Config'] == 'Narrative Union')]['f1 samples'].values[0]
})

df_plot2 = pd.DataFrame(plot2_data)

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))

# Define colors for plot 2
colors_plot2 = {
    'Vanilla': '#66c2a5',
    'Full Intersection': '#8da0cb',
    'Full Union': '#fc8d62',
    'Full Mixed': '#e78ac3',
    'Narrative Intersection': '#ffd92f',
    'Subnarrative Intersection': '#a6d854',
    'Narrative Union': '#b3b3b3'
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
    
    # Calculate positions for bars based on number of strategies
    num_strategies = len(strategies)
    if num_strategies == 4:  # EN or RU with 4 strategies
        width_p2 = 0.15
        positions = [lang_idx - 0.3, lang_idx - 0.1, lang_idx + 0.1, lang_idx + 0.3]
    elif num_strategies == 3:  # HI with 3 strategies
        width_p2 = 0.2
        positions = [lang_idx - 0.25, lang_idx, lang_idx + 0.25]
    else:  # Default 2 strategies
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
ax.set_title('Multi-Agent Strategies Across Languages', fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x_plot2)
ax.set_xticklabels(languages_plot2, fontsize=12)

# Create custom legend with all strategies in order
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors_plot2['Vanilla'], label='Vanilla', alpha=0.85, edgecolor='white', linewidth=1.5),
    Patch(facecolor=colors_plot2['Full Intersection'], label='Full Intersection', alpha=0.85, edgecolor='white', linewidth=1.5),
    Patch(facecolor=colors_plot2['Narrative Intersection'], label='Narrative Intersection', alpha=0.85, edgecolor='white', linewidth=1.5),
    Patch(facecolor=colors_plot2['Subnarrative Intersection'], label='Subnarrative Intersection', alpha=0.85, edgecolor='white', linewidth=1.5),
    Patch(facecolor=colors_plot2['Full Union'], label='Full Union', alpha=0.85, edgecolor='white', linewidth=1.5),
    Patch(facecolor=colors_plot2['Narrative Union'], label='Narrative Union', alpha=0.85, edgecolor='white', linewidth=1.5),
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
print(f"  EN Narrative Intersection: {df_plot2[(df_plot2['Language']=='EN') & (df_plot2['Strategy']=='Narrative Intersection')]['f1_samples'].values[0] - df_plot2[(df_plot2['Language']=='EN') & (df_plot2['Strategy']=='Vanilla')]['f1_samples'].values[0]:+.3f}")
print(f"  EN Subnarrative Intersection: {df_plot2[(df_plot2['Language']=='EN') & (df_plot2['Strategy']=='Subnarrative Intersection')]['f1_samples'].values[0] - df_plot2[(df_plot2['Language']=='EN') & (df_plot2['Strategy']=='Vanilla')]['f1_samples'].values[0]:+.3f}")
print(f"  HI Full Intersection: {df_plot2[(df_plot2['Language']=='HI') & (df_plot2['Strategy']=='Full Intersection')]['f1_samples'].values[0] - df_plot2[(df_plot2['Language']=='HI') & (df_plot2['Strategy']=='Vanilla')]['f1_samples'].values[0]:+.3f}")
print(f"  HI Narrative Intersection: {df_plot2[(df_plot2['Language']=='HI') & (df_plot2['Strategy']=='Narrative Intersection')]['f1_samples'].values[0] - df_plot2[(df_plot2['Language']=='HI') & (df_plot2['Strategy']=='Vanilla')]['f1_samples'].values[0]:+.3f}")
print(f"  RU Full Union: {df_plot2[(df_plot2['Language']=='RU') & (df_plot2['Strategy']=='Full Union')]['f1_samples'].values[0] - df_plot2[(df_plot2['Language']=='RU') & (df_plot2['Strategy']=='Vanilla')]['f1_samples'].values[0]:+.3f}")
print(f"  RU Narrative Union: {df_plot2[(df_plot2['Language']=='RU') & (df_plot2['Strategy']=='Narrative Union')]['f1_samples'].values[0] - df_plot2[(df_plot2['Language']=='RU') & (df_plot2['Strategy']=='Vanilla')]['f1_samples'].values[0]:+.3f}")
print(f"  RU Full Mixed: {df_plot2[(df_plot2['Language']=='RU') & (df_plot2['Strategy']=='Full Mixed')]['f1_samples'].values[0] - df_plot2[(df_plot2['Language']=='RU') & (df_plot2['Strategy']=='Vanilla')]['f1_samples'].values[0]:+.3f}")

plt.close()
