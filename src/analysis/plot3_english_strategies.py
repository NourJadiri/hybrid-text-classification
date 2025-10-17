"""
Plot 3: All English Classification Strategies
Comprehensive comparison of all 10 English configurations
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

# Filter all English configs
df_plot3 = df[df['Language'] == 'EN'].copy()

# Sort by f1 samples
df_plot3 = df_plot3.sort_values('f1 samples', ascending=True)

# Define colors for different strategy types
strategy_colors = {
    'Vanilla': '#66c2a5',
    'Narrative Intersection': '#8da0cb',
    'Narrative Union': '#fc8d62',
    'Narrative Majority': '#e78ac3',
    'Subnarrative': '#a6d854',
    'Full': '#ffd92f'
}

# Assign colors based on strategy type
colors_plot3 = []
strategy_types = []
for config in df_plot3['Config']:
    if 'Vanilla' in config:
        colors_plot3.append(strategy_colors['Vanilla'])
        strategy_types.append('Vanilla')
    elif 'Majority' in config:
        colors_plot3.append(strategy_colors['Narrative Majority'])
        strategy_types.append('Narrative Majority')
    elif 'Narrative Intersection' in config:
        colors_plot3.append(strategy_colors['Narrative Intersection'])
        strategy_types.append('Narrative Intersection')
    elif 'Subnarrative' in config:
        colors_plot3.append(strategy_colors['Subnarrative'])
        strategy_types.append('Subnarrative')
    elif 'Full' in config:
        colors_plot3.append(strategy_colors['Full'])
        strategy_types.append('Full')
    else:
        colors_plot3.append('#cccccc')
        strategy_types.append('Other')

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Create horizontal bar plot
y_pos = np.arange(len(df_plot3))
bars = ax.barh(y_pos, df_plot3['f1 samples'], color=colors_plot3, alpha=0.85, edgecolor='white', linewidth=1.5)

# Add value labels
for i, v in enumerate(df_plot3['f1 samples'].values):
    ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(df_plot3['Config'], fontsize=10)
ax.set_xlabel('F1 Samples', fontsize=13, fontweight='bold')
ax.set_title('All English Classification Strategies', fontsize=15, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_xlim(0, 0.6)
ax.set_axisbelow(True)

# No legend needed - configuration names are already shown on y-axis

plt.tight_layout()
plt.savefig('results/openai_gpt-5-nano/plot3_english_strategies.png', dpi=300, bbox_inches='tight')
print("Plot 3 saved:")
print("  - results/openai_gpt-5-nano/plot3_english_strategies.png")

# Summary statistics
print("\nTop 5 English strategies:")
for i, row in df_plot3.sort_values('f1 samples', ascending=False).head(5).iterrows():
    print(f"  {row['Config']}: F1 Samples = {row['f1 samples']:.3f}")

print("\nStrategy type breakdown:")
df_plot3['Strategy Type'] = strategy_types
for stype in ['Vanilla', 'Narrative Intersection', 'Narrative Majority', 'Subnarrative', 'Full']:
    count = len(df_plot3[df_plot3['Strategy Type'] == stype])
    if count > 0:
        avg_f1 = df_plot3[df_plot3['Strategy Type'] == stype]['f1 samples'].mean()
        print(f"  {stype}: {count} config(s), avg F1 Samples = {avg_f1:.3f}")

plt.close()
