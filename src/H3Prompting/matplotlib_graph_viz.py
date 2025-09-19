import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Define colors
start_end_color = '#4CAF50'
process_color = '#2196F3'
decision_color = '#FF9800'
handle_color = '#9C27B0'
arrow_color = '#333333'

# Node positions (x, y)
nodes = {
    'START': (5, 11),
    'categories': (5, 9.5),
    'narratives': (3, 8),
    'handle_other_category': (7, 8),
    'validate_narratives': (3, 6.5),
    'handle_empty_narratives': (1, 5),
    'clean_narratives': (3, 5),
    'subnarratives': (3, 3.5),
    'clean_subnarratives': (3, 2),
    'write_results': (5, 0.5),
    'END': (5, -0.5)
}

# Node styles
node_styles = {
    'START': {'color': start_end_color, 'shape': 'circle'},
    'END': {'color': start_end_color, 'shape': 'circle'},
    'categories': {'color': decision_color, 'shape': 'diamond'},
    'narratives': {'color': process_color, 'shape': 'rect'},
    'handle_other_category': {'color': handle_color, 'shape': 'rect'},
    'validate_narratives': {'color': decision_color, 'shape': 'diamond'},
    'handle_empty_narratives': {'color': handle_color, 'shape': 'rect'},
    'clean_narratives': {'color': process_color, 'shape': 'rect'},
    'subnarratives': {'color': process_color, 'shape': 'rect'},
    'clean_subnarratives': {'color': process_color, 'shape': 'rect'},
    'write_results': {'color': process_color, 'shape': 'rect'}
}

# Function to calculate edge points for nodes to avoid clipping
def get_node_edge_point(node_name, from_x, from_y, to_x, to_y):
    """Calculate the point on the edge of a node for arrow connection"""
    x, y = nodes[node_name]
    style = node_styles[node_name]
    
    # Calculate direction vector
    dx = to_x - from_x
    dy = to_y - from_y
    length = np.sqrt(dx**2 + dy**2)
    if length == 0:
        return x, y
    
    # Normalize direction
    dx_norm = dx / length
    dy_norm = dy / length
    
    # Calculate offset based on node shape
    if style['shape'] == 'circle':
        offset = 0.35  # Circle radius + small margin
    elif style['shape'] == 'diamond':
        offset = 0.45  # Diamond radius + small margin
    else:  # rectangle
        # For rectangles, calculate based on direction
        if abs(dx_norm) > abs(dy_norm):  # Horizontal connection
            offset = 0.65  # Half width + margin
        else:  # Vertical connection
            offset = 0.3   # Half height + margin
    
    return x + dx_norm * offset, y + dy_norm * offset

# Function to draw nodes
def draw_node(name, x, y, style):
    if style['shape'] == 'circle':
        circle = patches.Circle((x, y), 0.3, color=style['color'], alpha=0.8, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center', fontsize=10, fontweight='bold', zorder=3)
    elif style['shape'] == 'diamond':
        # Diamond shape for decision nodes
        diamond = patches.RegularPolygon((x, y), 4, radius=0.4, orientation=np.pi/4, 
                                       facecolor=style['color'], alpha=0.8, zorder=2)
        ax.add_patch(diamond)
        # Split long text into multiple lines
        text_lines = name.replace('_', '\n')
        ax.text(x, y, text_lines, ha='center', va='center', fontsize=9, fontweight='bold', zorder=3)
    else:  # rectangle
        # Split long text into multiple lines for better readability
        text_lines = name.replace('_', '\n')
        rect = FancyBboxPatch((x-0.6, y-0.25), 1.2, 0.5, boxstyle="round,pad=0.05",
                            facecolor=style['color'], alpha=0.8, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y, text_lines, ha='center', va='center', fontsize=9, fontweight='bold', zorder=3)

# Draw all nodes
for name, (x, y) in nodes.items():
    draw_node(name, x, y, node_styles[name])

# Function to draw arrows with orthogonal routing (horizontal/vertical only)
def draw_arrow(start_node, end_node, label='', offset=0, style='solid'):
    start_x, start_y = nodes[start_node]
    end_x, end_y = nodes[end_node]
    
    if offset != 0:
        # For multiple arrows between same nodes, create orthogonal paths with offset
        # Calculate intermediate points for orthogonal routing
        if abs(end_x - start_x) > abs(end_y - start_y):
            # Primarily horizontal movement
            mid_x = start_x + (end_x - start_x) * 0.5
            mid_y = start_y + offset * 0.8  # Apply offset vertically
            
            # Create path: start -> (mid_x, start_y) -> (mid_x, mid_y) -> (mid_x, end_y) -> end
            path_points = [
                (start_x, start_y),
                (mid_x, start_y),
                (mid_x, mid_y),
                (mid_x, end_y),
                (end_x, end_y)
            ]
        else:
            # Primarily vertical movement
            mid_x = start_x + offset * 0.8  # Apply offset horizontally
            mid_y = start_y + (end_y - start_y) * 0.5
            
            # Create path: start -> (start_x, mid_y) -> (mid_x, mid_y) -> (end_x, mid_y) -> end
            path_points = [
                (start_x, start_y),
                (start_x, mid_y),
                (mid_x, mid_y),
                (end_x, mid_y),
                (end_x, end_y)
            ]
        
        # Draw the orthogonal path
        for i in range(len(path_points) - 1):
            start_pt = path_points[i]
            end_pt = path_points[i + 1]
            
            # Skip zero-length segments
            if start_pt[0] == end_pt[0] and start_pt[1] == end_pt[1]:
                continue
                
            # Add arrowhead only on the last segment
            arrowstyle = '->' if i == len(path_points) - 2 else '-'
            
            ax.annotate('', xy=end_pt, xytext=start_pt,
                       arrowprops=dict(arrowstyle=arrowstyle, color=arrow_color, lw=1.5, linestyle=style),
                       zorder=1)
        
        # Add label at the midpoint
        if label:
            label_x = (path_points[0][0] + path_points[-1][0]) / 2
            label_y = (path_points[0][1] + path_points[-1][1]) / 2 + offset * 0.3
            ax.text(label_x, label_y, label, ha='center', va='center', fontsize=7,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    else:
        # For straight connections, use orthogonal routing
        if start_x == end_x:
            # Pure vertical connection
            start_edge_x, start_edge_y = get_node_edge_point(start_node, start_x, start_y, end_x, end_y)
            end_edge_x, end_edge_y = get_node_edge_point(end_node, end_x, end_y, start_x, start_y)
            
            ax.annotate('', xy=(end_edge_x, end_edge_y), xytext=(start_edge_x, start_edge_y),
                       arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5, linestyle=style),
                       zorder=1)
        elif start_y == end_y:
            # Pure horizontal connection
            start_edge_x, start_edge_y = get_node_edge_point(start_node, start_x, start_y, end_x, end_y)
            end_edge_x, end_edge_y = get_node_edge_point(end_node, end_x, end_y, start_x, start_y)
            
            ax.annotate('', xy=(end_edge_x, end_edge_y), xytext=(start_edge_x, start_edge_y),
                       arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5, linestyle=style),
                       zorder=1)
        else:
            # Mixed horizontal/vertical - create L-shaped path
            if abs(end_x - start_x) > abs(end_y - start_y):
                # Horizontal first, then vertical
                corner_x = end_x
                corner_y = start_y
                
                # Draw horizontal segment
                start_edge_x, start_edge_y = get_node_edge_point(start_node, start_x, start_y, corner_x, corner_y)
                ax.annotate('', xy=(corner_x, corner_y), xytext=(start_edge_x, start_edge_y),
                           arrowprops=dict(arrowstyle='-', color=arrow_color, lw=1.5, linestyle=style),
                           zorder=1)
                
                # Draw vertical segment with arrow
                end_edge_x, end_edge_y = get_node_edge_point(end_node, end_x, end_y, corner_x, corner_y)
                ax.annotate('', xy=(end_edge_x, end_edge_y), xytext=(corner_x, corner_y),
                           arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5, linestyle=style),
                           zorder=1)
            else:
                # Vertical first, then horizontal
                corner_x = start_x
                corner_y = end_y
                
                # Draw vertical segment
                start_edge_x, start_edge_y = get_node_edge_point(start_node, start_x, start_y, corner_x, corner_y)
                ax.annotate('', xy=(corner_x, corner_y), xytext=(start_edge_x, start_edge_y),
                           arrowprops=dict(arrowstyle='-', color=arrow_color, lw=1.5, linestyle=style),
                           zorder=1)
                
                # Draw horizontal segment with arrow
                end_edge_x, end_edge_y = get_node_edge_point(end_node, end_x, end_y, corner_x, corner_y)
                ax.annotate('', xy=(end_edge_x, end_edge_y), xytext=(corner_x, corner_y),
                           arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5, linestyle=style),
                           zorder=1)
        
        # Add label if provided
        if label:
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            ax.text(mid_x, mid_y, label, ha='center', va='center', fontsize=7,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

# Define edges based on the graph.py structure
edges = [
    # Main flow
    ('START', 'categories'),
    
    # From categories (conditional)
    ('categories', 'narratives', 'category != "Other"'),
    ('categories', 'handle_other_category', 'category == "Other"', 0.3),
    
    # From narratives (conditional)
    ('narratives', 'validate_narratives', 'narratives found'),
    ('narratives', 'handle_empty_narratives', 'no narratives', 0.4),
    
    # From validate_narratives (conditional - retry loop)
    ('validate_narratives', 'clean_narratives', 'approved or max retries'),
    ('validate_narratives', 'narratives', 'retry needed', 0.5, 'dashed'),
    
    # From clean_narratives (conditional)
    ('clean_narratives', 'subnarratives', 'narratives remain'),
    ('clean_narratives', 'handle_empty_narratives', 'no narratives after cleaning', 0.3),
    
    # Linear flow
    ('subnarratives', 'clean_subnarratives'),
    
    # All paths converge to write_results
    ('handle_other_category', 'write_results'),
    ('handle_empty_narratives', 'write_results'),
    ('clean_subnarratives', 'write_results'),
    
    # Final step
    ('write_results', 'END')
]

# Draw all edges
for edge in edges:
    if len(edge) == 2:
        draw_arrow(edge[0], edge[1])
    elif len(edge) == 3:
        draw_arrow(edge[0], edge[1], edge[2])
    elif len(edge) == 4:
        draw_arrow(edge[0], edge[1], edge[2], edge[3])
    elif len(edge) == 5:
        draw_arrow(edge[0], edge[1], edge[2], edge[3], edge[4])

# Add title and legend
plt.title('Text Classification Workflow Graph', fontsize=16, fontweight='bold', pad=20)

# Create legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=start_end_color, 
               markersize=10, label='Start/End'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor=decision_color, 
               markersize=10, label='Decision Points'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=process_color, 
               markersize=10, label='Processing Steps'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=handle_color, 
               markersize=10, label='Error Handlers'),
    Line2D([0], [0], color=arrow_color, linewidth=2, label='Normal Flow'),
    Line2D([0], [0], color=arrow_color, linewidth=2, linestyle='--', label='Retry Flow')
]

ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

# Add workflow description
description = """
Workflow Description:
1. START → Classify text into categories
2. If "Other" category → Handle directly
3. If valid category → Classify narratives
4. Validate narratives (with retry mechanism)
5. Clean invalid narratives
6. Classify subnarratives
7. Clean invalid subnarratives
8. Write results → END
"""

ax.text(0.1, 3, description, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))

# Adjust layout and save
plt.tight_layout()
plt.savefig('classification_workflow_graph.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('classification_workflow_graph.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print("✅ Graph visualization saved as:")
print("   - classification_workflow_graph.png (high-resolution PNG)")
print("   - classification_workflow_graph.pdf (vector PDF)")
print("\n📊 Graph Statistics:")
print(f"   - Total Nodes: {len(nodes)}")
print(f"   - Total Edges: {len(edges)}")
print(f"   - Decision Points: {sum(1 for style in node_styles.values() if style['shape'] == 'diamond')}")
print(f"   - Process Steps: {sum(1 for style in node_styles.values() if style['shape'] == 'rect')}")

plt.show()