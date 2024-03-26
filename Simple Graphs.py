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

    # Calculate outgoing edge weights for each node
    out_edge_weights = {}
    for node in nodes:
        out_edge_weights[node] = sum([G[node][v]['weight'] for v in G.neighbors(node)])

    # Customize node shapes and sizes
    node_patches = [patches.Rectangle((0, 0), 0.05, out_edge_weights[node]/50, edgecolor='black', facecolor='skyblue') for node in nodes]

    # Specify positions of the nodes
    pos = {'A': (-3, 1), 'B': (-2, -2), 'C': (0.5, 0), 'D': (0, -1), 'E': (2, 1), 'F': (3, -2)}

    # Draw the graph with specified node positions
    nx.draw(G, pos, with_labels=True, width=[G[u][v]['weight'] for u, v in G.edges()], edge_color='b', node_size=1, alpha=0.7)

    # Add node patches
    for i, patch in enumerate(node_patches):
        x, y = pos[nodes[i]]
        patch.set_xy([x, y])
        plt.gca().add_patch(patch)

    # Show the plot
    plt.show()

# Example usage:
nodes = ['A', 'B', 'C', 'D', 'E', 'F']
edges = np.array([[0, 0, 2, 5, 0, 0], [0, 0, 4, 1, 0, 1], [0, 0, 0, 0, 3, 3], [0, 0, 0, 0, 2, 4], [0,0,0,0,0,0], [0,0,0,0,0,0]])  # Example adjacency matrix

draw_weighted_graph(nodes, edges)
