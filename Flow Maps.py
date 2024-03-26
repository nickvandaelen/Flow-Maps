# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:00:28 2024

@author: 18582
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_weighted_graph(nodes, edges):
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

    # Calculate outgoing and incoming edge weights for each node
    out_edge_weights = {}
    in_edge_weights = {}
    for i in range(len(nodes)):
        out_edge_weights[nodes[i]] = sum(edges[i][:])
        in_edge_weights[nodes[i]] = sum(np.transpose(edges)[i])
    print(out_edge_weights)
    print(in_edge_weights)

    # Customize node shapes and sizes
    #node_height = min(out_edge_weights[node], out_edge_weights[node])
    node_patches = [patches.Rectangle((0, 0), 0.07, max(out_edge_weights[node], in_edge_weights[node])/50, edgecolor='black', facecolor='skyblue') for node in nodes]

    # Specify positions of the nodes
    pos = {'A': (-3, 1), 'B': (-2, -2), 'C': (0.5, 0), 'D': (0, -1), 'E': (2, 1), 'F': (3, -2)}

    # Draw the graph with specified node positions
    #nx.draw(G, pos, with_labels=True, width=[G[u][v]['weight'] * 2 for u, v in G.edges()], edge_color='b', node_size=1, alpha=0.7)

    # Draw edges with custom routing
    for edge in G.edges():
        node1, node2 = edge
        weight = G.edges[edge]['weight']  # Get the weight of the current edge
        x1, y1 = pos[node1]
        x2, y2 = pos[node2]
        control_point = (x1 + 0.3, y1)
        control_point_2 = (x2 - 0.3, y2) # Control point for the Bezier curve

        plt.plot([x1+out_edge_weights[node1]/50, control_point[0]], [y1, control_point[1]], 'k-', linewidth=out_edge_weights[node1])  # Horizontal line
        plt.plot([control_point[0], control_point_2[0]], [control_point[1], control_point_2[1]], 'k-', linewidth=weight)  # Bezier curve
        plt.plot([control_point_2[0], x2-in_edge_weights[node2]/80], [control_point_2[1], y2], 'k-', linewidth=in_edge_weights[node2])  #

        
            


    # Add node patches
    for i, patch in enumerate(node_patches):
        x, y = pos[nodes[i]]
        patch.set_xy([x-0.01, y-0.03])
        plt.gca().add_patch(patch)

    # Show the plot
    plt.show()

# Example usage:
nodes = ['A', 'B', 'C', 'D', 'E', 'F']
edges = np.array([[0, 0, 2, 5, 0, 0], [0, 0, 1, 2, 0, 0], [0, 0, 0, 0, 8, 3], [0, 0, 0, 0, 2, 4], [0,0,0,0,0,0], [0,0,0,0,0,0]])  # Example adjacency matrix

draw_weighted_graph(nodes, edges)
