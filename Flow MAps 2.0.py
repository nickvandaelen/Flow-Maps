import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm

def draw_weighted_graph(nodes, edges):
    # Create a graph
    G = nx.DiGraph()

    # Add nodes to the graph
    for node in nodes:
        G.add_node(node)

    # Add edges to the graph with weights
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            weight = edges[i][j]
            if weight != 0:
                G.add_edge(nodes[i], nodes[j], weight=weight)

    # Calculate outgoing and incoming edge weights for each node
    out_edge_weights = {}
    in_edge_weights = {}
    for node in nodes:
        out_edge_weights[node] = sum(G.edges[edge]['weight'] for edge in G.edges() if edge[0] == node)
        in_edge_weights[node] = sum(G.edges[edge]['weight'] for edge in G.edges() if edge[1] == node)

    # Create a colormap
    cmap = cm.get_cmap('viridis')

    # Specify positions of the nodes
    pos = {'A': (-3, 1), 'B': (-2, -2), 'C': (0.5, 0), 'D': (0, -1), 'E': (2, 1), 'F': (3, -2), 'J':(-0.75, -1.5)}

    # Extract x-values from the points
    x_values = [point[0] for point in pos.values()]

    # Set the x-limits of the plot
    plt.xlim(min(x_values) - 1, max(x_values) + 1)
    
    #linewidth Scaling Factor
    #fact needs to be 600/width of graph
    fact = 600/(max(x_values) - min(x_values))

    # Draw edges with custom routing and color
    running_in_width = {node: 0 for node in nodes}
    running_out_width = {node: 0 for node in nodes}
    for edge in G.edges():
        node1, node2 = edge
        
        weight = G.edges[edge]['weight']  # Get the weight of the current edge
        x1, y1 = pos[node1]
        x2, y2 = pos[node2]
        seg = 0.5

        control_point = (x1 + seg, y1 + out_edge_weights[node1]/fact-running_out_width[node1]/(fact/2)-weight/fact)
        control_point_2 = (x2 - seg, y2 + in_edge_weights[node2]/fact-running_in_width[node2]/(fact/2)-weight/fact) # Control point for the Bezier curve

        # Map the x-coordinate of the starting point to a color value
        color_value = (y1 - min(pos[node][1] for node in nodes)) / (max(pos[node][1] for node in nodes) - min(pos[node][1] for node in nodes))
        color = cmap(color_value)

        # Draw the first segment of the edge with width equal to the weight of the current edge
        plt.plot([x1, control_point[0]], [y1+ out_edge_weights[node1]/fact-running_out_width[node1]/(fact/2)-weight/fact, control_point[1]], color=color, linewidth=weight, alpha=min(1/weight+0.5, 1), solid_capstyle="butt")  # Horizontal line

        # Draw the Bezier curve segment with width equal to the weight of the current edge
        plt.plot([control_point[0], control_point_2[0]], [control_point[1], control_point_2[1]], color=color, linewidth=weight, alpha=min(1/weight+0.5, 1), solid_capstyle="round")  # Bezier curve

        # Draw the last segment of the edge with width equal to the weight of the current edge
        plt.plot([control_point_2[0], x2], [control_point_2[1], y2+ in_edge_weights[node2]/fact-running_in_width[node2]/(fact/2)-weight/fact], color=color, linewidth=weight, alpha=min(1/weight+0.5, 1), solid_capstyle="butt")  #

        running_out_width[node1] += weight
        running_in_width[node2] += weight
        
        print(edge)
    # Customize node shapes and sizes
    node_patches = [patches.Circle((0, 0), max(out_edge_weights[node], in_edge_weights[node])/50, edgecolor='black', facecolor='black') for node in nodes]

    # Add node patches
    for i, patch in enumerate(node_patches):
        x, y = pos[nodes[i]]
        patch.center = (x, y)
        plt.gca().add_patch(patch)
        
    # Show the plot
    plt.axis('equal')
    plt.axis('off')
    plt.show()

# Example usage:
nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'J']
edges = np.array([[0, 0, 2, 5, 0, 0, 0], 
                  [0, 0, 1, 2, 1, 0, 5], 
                  [0, 0, 0, 0, 8, 3, 0], 
                  [0, 0, 0, 0, 2, 4, 0], 
                  [0, 0, 0, 0, 0, 0, 0], 
                  [0, 0, 0, 0, 0, 0, 0], 
                  [0, 0, 0, 0, 3, 2, 0]])  # Example adjacency matrix

'''
#test if nodes are different order
nodes = ['A', 'C', 'B',  'D', 'E', 'F', 'J']
edges = np.array([[0, 2, 0, 5, 0, 0], [0, 0, 0, 0, 8, 3], [0, 0, 1, 2, 1, 0],  [0, 0, 0, 0, 2, 4], [0,0,0,0,0,0], [0,0,0,0,0,0]])  # Example adjacency matrix

    # Create a graph
G = nx.Graph()

    # Add nodes to the graph
for node in nodes:
    G.add_node(node)

    # Add edges to the graph with weights
for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)):
        weight = edges[i][j]
        if weight != 0:
            G.add_edge(nodes[i], nodes[j], weight=weight)
for edge in G.edges():
    print(edge)
'''
draw_weighted_graph(nodes, edges)
