"""
Create three plots analyzing multi-agent classification strategies.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 5)
plt.rcParams['font.size'] = 10

# Read the data
df = pd.read_csv('results/openai_gpt-5-nano/summary.csv')

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# ========== PLOT 1: Basic Narrative Strategies (All Languages) ==========
print("Creating Plot 1: Basic Narrative Strategies...")

# Filter for vanilla, narrative intersection, and narrative union
configs_plot1 = ['Vanilla', 'Narrative Intersection', 'Narrative Union']
df_plot1 = df[df['Config'].isin(configs_plot1)].copy()

# Create grouped bar plot
languages = ['BG', 'EN', 'HI', 'PT', 'RU']
x = np.arange(len(languages))
width = 0.25

# Define colors for each strategy
colors = {
    'Vanilla': '#7fc97f',
    'Narrative Intersection': '#beaed4',
    'Narrative Union': '#fdc086'
}

# Plot F1 macro coarse
for i, config in enumerate(configs_plot1):
    values = []
    for lang in languages:
        val = df_plot1[(df_plot1['Language'] == lang) & (df_plot1['Config'] == config)]['f1 macro coarse']
        values.append(val.values[0] if len(val) > 0 else 0)
    
    axes[0].bar(x + i*width, values, width, label=config, color=colors[config], alpha=0.8)

axes[0].set_xlabel('Language', fontsize=12, fontweight='bold')
axes[0].set_ylabel('F1 Macro Coarse', fontsize=12, fontweight='bold')
axes[0].set_title('Plot 1: Basic Narrative Strategies Across Languages', fontsize=14, fontweight='bold')
axes[0].set_xticks(x + width)
axes[0].set_xticklabels(languages)
axes[0].legend(loc='upper left', framealpha=0.9)
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_ylim([0, 0.7])

# Add value labels on bars
for i, config in enumerate(configs_plot1):
    values = []
    for lang in languages:
        val = df_plot1[(df_plot1['Language'] == lang) & (df_plot1['Config'] == config)]['f1 macro coarse']
        values.append(val.values[0] if len(val) > 0 else 0)
    
    for j, v in enumerate(values):
        if v > 0:
            axes[0].text(x[j] + i*width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

# ========== PLOT 2: Advanced Strategies (Full Intersection/Union/Mixed) ==========
print("Creating Plot 2: Advanced Strategies...")

# Prepare data for plot 2
plot2_data = []

# English Full Intersection vs Vanilla
plot2_data.append({
    'Language': 'EN',
    'Strategy': 'Vanilla',
    'f1_macro': df[(df['Language'] == 'EN') & (df['Config'] == 'Vanilla')]['f1 macro coarse'].values[0]
})
plot2_data.append({
    'Language': 'EN',
    'Strategy': 'Full Intersection',
    'f1_macro': df[(df['Language'] == 'EN') & (df['Config'] == 'Full Intersection')]['f1 macro coarse'].values[0]
})

# Hindi Full Intersection vs Vanilla
plot2_data.append({
    'Language': 'HI',
    'Strategy': 'Vanilla',
    'f1_macro': df[(df['Language'] == 'HI') & (df['Config'] == 'Vanilla')]['f1 macro coarse'].values[0]
})
plot2_data.append({
    'Language': 'HI',
    'Strategy': 'Full Intersection',
    'f1_macro': df[(df['Language'] == 'HI') & (df['Config'] == 'Full Intersection')]['f1 macro coarse'].values[0]
})

# Russian Full Union, Full Mixed vs Vanilla
plot2_data.append({
    'Language': 'RU',
    'Strategy': 'Vanilla',
    'f1_macro': df[(df['Language'] == 'RU') & (df['Config'] == 'Vanilla')]['f1 macro coarse'].values[0]
})
plot2_data.append({
    'Language': 'RU',
    'Strategy': 'Full Union',
    'f1_macro': df[(df['Language'] == 'RU') & (df['Config'] == 'Full Union')]['f1 macro coarse'].values[0]
})
plot2_data.append({
    'Language': 'RU',
    'Strategy': 'Full Mixed',
    'f1_macro': df[(df['Language'] == 'RU') & (df['Config'] == 'Full Mixed')]['f1 macro coarse'].values[0]
})

df_plot2 = pd.DataFrame(plot2_data)

# Define colors for plot 2
colors_plot2 = {
    'Vanilla': '#7fc97f',
    'Full Intersection': '#386cb0',
    'Full Union': '#f0027f',
    'Full Mixed': '#bf5b17'
}

# Group by language
languages_plot2 = ['EN', 'HI', 'RU']
x_plot2 = np.arange(len(languages_plot2))

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
        value = lang_data[lang_data['Strategy'] == strategy]['f1_macro'].values[0]
        axes[1].bar(positions[i], value, width_p2, label=strategy if lang_idx == 0 or (lang == 'RU' and strategy == 'Full Mixed') else '', 
                   color=colors_plot2[strategy], alpha=0.8)
        axes[1].text(positions[i], value + 0.01, f'{value:.3f}', ha='center', va='bottom', fontsize=8)

axes[1].set_xlabel('Language', fontsize=12, fontweight='bold')
axes[1].set_ylabel('F1 Macro Coarse', fontsize=12, fontweight='bold')
axes[1].set_title('Plot 2: Advanced Multi-Agent Strategies', fontsize=14, fontweight='bold')
axes[1].set_xticks(x_plot2)
axes[1].set_xticklabels(languages_plot2)
axes[1].legend(loc='upper left', framealpha=0.9)
axes[1].grid(axis='y', alpha=0.3)
axes[1].set_ylim([0, 0.7])

# ========== PLOT 3: All English Configurations ==========
print("Creating Plot 3: All English Configurations...")

# Filter all English configs
df_plot3 = df[df['Language'] == 'EN'].copy()

# Sort by f1 macro coarse
df_plot3 = df_plot3.sort_values('f1 macro coarse', ascending=False)

# Define colors for different strategy types
strategy_colors = {
    'Vanilla': '#7fc97f',
    'Narrative Intersection': '#beaed4',
    'Narrative Union': '#fdc086',
    'Narrative Majority': '#ffff99',
    'Subnarrative': '#fb8072',
    'Full': '#386cb0'
}

# Assign colors based on strategy type
colors_plot3 = []
for config in df_plot3['Config']:
    if 'Vanilla' in config:
        colors_plot3.append(strategy_colors['Vanilla'])
    elif 'Majority' in config:
        colors_plot3.append(strategy_colors['Narrative Majority'])
    elif 'Narrative Intersection' in config:
        colors_plot3.append(strategy_colors['Narrative Intersection'])
    elif 'Subnarrative' in config:
        colors_plot3.append(strategy_colors['Subnarrative'])
    elif 'Full' in config:
        colors_plot3.append(strategy_colors['Full'])
    else:
        colors_plot3.append('#cccccc')

# Create horizontal bar plot
y_pos = np.arange(len(df_plot3))
axes[2].barh(y_pos, df_plot3['f1 macro coarse'], color=colors_plot3, alpha=0.8)

# Add value labels
for i, v in enumerate(df_plot3['f1 macro coarse'].values):
    axes[2].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=8)

axes[2].set_yticks(y_pos)
axes[2].set_yticklabels(df_plot3['Config'], fontsize=9)
axes[2].set_xlabel('F1 Macro Coarse', fontsize=12, fontweight='bold')
axes[2].set_title('Plot 3: All English Classification Strategies', fontsize=14, fontweight='bold')
axes[2].grid(axis='x', alpha=0.3)
axes[2].set_xlim([0, 0.6])

# Add legend for strategy types
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=strategy_colors['Vanilla'], label='Vanilla', alpha=0.8),
    Patch(facecolor=strategy_colors['Narrative Intersection'], label='Narrative Intersection', alpha=0.8),
    Patch(facecolor=strategy_colors['Narrative Majority'], label='Narrative Majority', alpha=0.8),
    Patch(facecolor=strategy_colors['Subnarrative'], label='Subnarrative', alpha=0.8),
    Patch(facecolor=strategy_colors['Full'], label='Full Intersection', alpha=0.8)
]
axes[2].legend(handles=legend_elements, loc='lower right', framealpha=0.9, fontsize=8)

# Adjust layout and save
plt.tight_layout()
plt.savefig('results/openai_gpt-5-nano/multi_strategy_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('results/openai_gpt-5-nano/multi_strategy_analysis.pdf', bbox_inches='tight')
print("\nPlots saved as:")
print("  - results/openai_gpt-5-nano/multi_strategy_analysis.png")
print("  - results/openai_gpt-5-nano/multi_strategy_analysis.pdf")
plt.show()

print("\n=== Summary Statistics ===")
print("\nPlot 1 - Best performing strategy per language:")
for lang in languages:
    lang_data = df_plot1[df_plot1['Language'] == lang]
    best = lang_data.loc[lang_data['f1 macro coarse'].idxmax()]
    print(f"  {lang}: {best['Config']} (F1 = {best['f1 macro coarse']:.3f})")

print("\nPlot 2 - Improvement from Vanilla:")
print(f"  EN Full Intersection: {df_plot2[(df_plot2['Language']=='EN') & (df_plot2['Strategy']=='Full Intersection')]['f1_macro'].values[0] - df_plot2[(df_plot2['Language']=='EN') & (df_plot2['Strategy']=='Vanilla')]['f1_macro'].values[0]:+.3f}")
print(f"  HI Full Intersection: {df_plot2[(df_plot2['Language']=='HI') & (df_plot2['Strategy']=='Full Intersection')]['f1_macro'].values[0] - df_plot2[(df_plot2['Language']=='HI') & (df_plot2['Strategy']=='Vanilla')]['f1_macro'].values[0]:+.3f}")
print(f"  RU Full Union: {df_plot2[(df_plot2['Language']=='RU') & (df_plot2['Strategy']=='Full Union')]['f1_macro'].values[0] - df_plot2[(df_plot2['Language']=='RU') & (df_plot2['Strategy']=='Vanilla')]['f1_macro'].values[0]:+.3f}")
print(f"  RU Full Mixed: {df_plot2[(df_plot2['Language']=='RU') & (df_plot2['Strategy']=='Full Mixed')]['f1_macro'].values[0] - df_plot2[(df_plot2['Language']=='RU') & (df_plot2['Strategy']=='Vanilla')]['f1_macro'].values[0]:+.3f}")

print("\nPlot 3 - Top 3 English strategies:")
for i, row in df_plot3.head(3).iterrows():
    print(f"  {row['Config']}: F1 = {row['f1 macro coarse']:.3f}")
