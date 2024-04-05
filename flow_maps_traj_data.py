import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from trajectory import *

def draw_weighted_graph(nodes, edges, positions):
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
    # pos = {'A': (-3, 1), 'B': (-2, -2), 'C': (0.5, 0), 'D': (0, -1), 'E': (2, 1), 'F': (3, -2), 'J':(-0.75, -1.5)}
    # pos = read_from_json(r"algorithm_repository\Flow-Maps\city_coordinates.json")
    pos = positions

    # Extract x-values from the points
    x_values = [point[0] for point in pos.values()]

    # Extract y-values from the points
    y_values = [point[1] for point in getDictValues(pos)]

    # Normalize
    x_values_norm = min_max_scaling(x_values, -5, 5)
    y_values_norm = min_max_scaling(y_values, -5, 5)
    pos_norm = dict()
    for i in range(len(x_values)):
        pos_norm[getDictKeys(pos)[i]] = (x_values_norm[i], y_values_norm[i])
    pos = pos_norm

    # Set the x-limits of the plot
    plt.xlim(min(x_values_norm) - 1, max(x_values_norm) + 1)

    # Set the y-limits of the plot
    plt.ylim(min(y_values_norm) - 1, max(y_values_norm) + 1)
    
    #linewidth Scaling Factor
    #fact needs to be 600/width of graph
    fact = 600/(max(x_values_norm) - min(x_values_norm))

    # Draw edges with custom routing and color
    running_in_width = {node: 0 for node in nodes}
    running_out_width = {node: 0 for node in nodes}
    for edge in G.edges():
        node1, node2 = edge
        
        weight = G.edges[edge]['weight']  # Get the weight of the current edge
        x1, y1 = pos[node1]
        x2, y2 = pos[node2]
        seg = abs(x1 - x2)/5

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
# nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'J']
# edges = np.array([[0, 0, 2, 5, 0, 0, 0], 
#                   [0, 0, 1, 2, 1, 0, 5], 
#                   [0, 0, 0, 0, 8, 3, 0], 
#                   [0, 0, 0, 0, 2, 4, 0], 
#                   [0, 0, 0, 0, 0, 0, 0], 
#                   [0, 0, 0, 0, 0, 0, 0], 
#                   [0, 0, 0, 0, 3, 2, 0]])  # Example adjacency matrix
# Get trajectories
t = Trajectory(r"algorithm_repository\Flow-Maps\trajectories.txt")
pos = read_from_json(r"algorithm_repository\Flow-Maps\city_coordinates.json")
nodes = t.unique_points
trajectories = t.filterOnStartAndEndPoints(['I'], ['G'])
edges = t.constructWeightMatrix(trajectories)
edges = (edges / np.max(edges)) * 8

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
draw_weighted_graph(nodes, edges, pos)

# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from matplotlib import cm
# from trajectory import *

# def draw_weighted_graph(nodes, edges, positions):
#     # Create a graph
#     G = nx.DiGraph()

#     # Add nodes to the graph
#     for node in nodes:
#         G.add_node(node)
        
        
#     # Specify positions of the nodes
#     pos = positions

#     # Extract x-values from the points
#     x_values = [point[0] for point in pos.values()]

#     # Set the x-limits of the plot
#     plt.xlim(min(x_values) - 1, max(x_values) + 1)

#     seg = (max(x_values) - min(x_values))/10   # Add edges to the graph with weights
#     for i in range(len(nodes)):
#         for j in range(len(nodes)):
#             weight = edges[i][j]
#             node1 = nodes[i]
#             node2 = nodes[j]
#             x1,y1 = pos[node1]
#             x2,y2 = pos[node2]
#             rise=(y2-y1)
#             run=(x2-x1)-1
#             if run != 0:
#                 dir_exit = np.arctan2(rise, (-1*run))
#                 dir_enter = np.arctan2(-rise, -run) 
#                 if dir_exit < 0:
#                     dir_exit += 2*np.pi
#                 if dir_enter < 0:
#                     dir_enter += 2*np.pi
#             elif y1 > y2:
#                 dir_exit = 3*np.pi/2
#                 dir_enter = np.pi/2
#             else:
#                 dir_exit = np.pi/2
#                 dir_enter = 3*np.pi/2
#             if weight != 0:
#                 G.add_edge(nodes[i], nodes[j], weight=weight, d1 = round(dir_exit,3), d2 = round(dir_enter,3))
    
#     #Calculate ROtational Positional order
#     for node in G.nodes():
#         # Get successor edges sorted by d1
#         successor_edges = sorted(G.edges(node, data=True), key=lambda x: x[2]['d1'])
#         # Initialize counters for p1 and p2
#         p1_counter = 0
#         p2_counter = 0

        
#         # Get predecessor edges sorted by d1
#         predecessor_edges = sorted(G.in_edges(node, data=True), key=lambda x: x[2]['d2'])
#         for u, v, edge_data in successor_edges:
#             # Assign p1 value to the edge
#             G.edges[u, v]['p1'] = p1_counter
#             # Increment p1 counter
#             p1_counter += 1
        
#         # Get predecessor edges sorted by d1
#         for u, v, edge_data in predecessor_edges:
#             # Assign p2 value to the edge
#             G.edges[u, v]['p2'] = p2_counter
#             # Increment p2 counter
#             p2_counter += 1



#     # Calculate outgoing and incoming edge weights for each node
#     out_edge_weights = {}
#     in_edge_weights = {}
#     for node in nodes:
#         out_edge_weights[node] = sum(G.edges[edge]['weight'] for edge in G.edges() if edge[0] == node)
#         in_edge_weights[node] = sum(G.edges[edge]['weight'] for edge in G.edges() if edge[1] == node)

#     # Create a colormap
#     cmap = cm.get_cmap('viridis')


#     # Extract x-values from the points
#     x_values = [point[0] for point in pos.values()]

#     # Set the x-limits of the plot
#     plt.xlim(min(x_values) - 1, max(x_values) + 1)
    

            
#     '''      
#             #### Now we have to insert the algorithm to reorder the edges correctly
#     edges = G.edges()
#     def sort_edges(edges):
#         new_edges = []
#         list_to_remove = []
#         items_placed_by_pass = []
#         iter_count = 0
#         p1_counters = {node: 0 for node in nodes}
#         p2_counters = {node: 0 for node in nodes}
#         while len(edges) > 0:
#             for i in range(len(edges)):
#                 if edges[i]['p1'] == p1_counters[edges[i]['node1']] and edges[i]['p2'] == p2_counters[edges[i]['node2']] :
#                     new_edges.append(edges[i])
#                     list_to_remove.append(i)
#                     p1_counters[edges[i]['node1']] += 1
#                     p2_counters[edges[i]['node2']] += 1
#             edges = [edges[i] for i in range(len(edges)) if i not in list_to_remove]
#             items_placed_by_pass.append(len(list_to_remove))
#             list_to_remove = []
#             iter_count += 1
#         return new_edges
            
#     edges = sort_edges(edges)
#     '''
#     #linewidth Scaling Factor
#     #fact needs to be 600/width of graph
#     fact = 600/(max(x_values) - min(x_values))

#     # Draw edges with custom routing and color
    
#     #initalize these dicts so we know how much to shift the line segments so they stack
#     running_in_width = {node: 0 for node in nodes}
#     running_out_width = {node: 0 for node in nodes}
    
#     ##Counters to determine if we can place a certain edge
#     p1_counters = {node: 0 for node in nodes}
#     p2_counters = {node: 0 for node in nodes}
#     ##Stopping mechanism
#     edges_placed = 0
#     while edges_placed < len(G.edges()):
#         # print(edges_placed, len(G.edges()))
#         for edge in G.edges():
#             node1, node2 = edge
#             if edges_placed == 0:
#                 print(node1, node2, G.edges[edge])
#             if G.edges[edge]['p1'] == p1_counters[node1] and G.edges[edge]['p2'] == p2_counters[node2]:
#                 print(node1, node2)
#                 weight = G.edges[edge]['weight']  # Get the weight of the current edge
#                 x1, y1 = pos[node1]
#                 x2, y2 = pos[node2]
#                 seg = seg
        
#                 control_point = (x1 + seg, y1 + out_edge_weights[node1]/fact-running_out_width[node1]/(fact/2)-weight/fact)
#                 control_point_2 = (x2 - seg, y2 + in_edge_weights[node2]/fact-running_in_width[node2]/(fact/2)-weight/fact) # Control point for the Bezier curve
        
#                 # Map the x-coordinate of the starting point to a color value
#                 color_value = (y1 - min(pos[node][1] for node in nodes)) / (max(pos[node][1] for node in nodes) - min(pos[node][1] for node in nodes))
#                 color = cmap(color_value)
        
#                 # Draw the first segment of the edge with width equal to the weight of the current edge
#                 plt.plot([x1, control_point[0]], [y1+ out_edge_weights[node1]/fact-running_out_width[node1]/(fact/2)-weight/fact, control_point[1]], color=color, linewidth=weight, alpha=min(1/weight+0.5, 1), solid_capstyle="butt")  # Horizontal line
        
#                 # Draw the Bezier curve segment with width equal to the weight of the current edge
#                 plt.plot([control_point[0], control_point_2[0]], [control_point[1], control_point_2[1]], color=color, linewidth=weight, alpha=min(1/weight+0.5, 1), solid_capstyle="round")  # Bezier curve
        
#                 # Draw the last segment of the edge with width equal to the weight of the current edge
#                 plt.plot([control_point_2[0], x2], [control_point_2[1], y2+ in_edge_weights[node2]/fact-running_in_width[node2]/(fact/2)-weight/fact], color=color, linewidth=weight, alpha=min(1/weight+0.5, 1), solid_capstyle="butt")  #
        
#                 running_out_width[node1] += weight
#                 running_in_width[node2] += weight
              
#                 p1_counters[node1] += 1
#                 p2_counters[node2] += 1
#                 edges_placed += 1
                
#     # Customize node shapes and sizes
#     node_patches = [patches.Circle((0, 0), max(out_edge_weights[node], in_edge_weights[node])/50, edgecolor='black', facecolor='black') for node in nodes]

#     # Add node patches
#     for i, patch in enumerate(node_patches):
#         x, y = pos[nodes[i]]
#         patch.center = (x, y)
#         plt.gca().add_patch(patch)
        
#     # Show the plot
#     plt.axis('equal')
#     plt.axis('off')
#     plt.show()


# # Get trajectories
# t = Trajectory(r"algorithm_repository\Flow-Maps\trajectories.txt")
# pos = read_from_json(r"algorithm_repository\Flow-Maps\city_coordinates.json")
# # Extract x-values from the points
# x_values = [point[0] for point in pos.values()]

# # Extract y-values from the points
# y_values = [point[1] for point in getDictValues(pos)]

# # Normalize
# x_values_norm = min_max_scaling(x_values, -100, 100)
# y_values_norm = min_max_scaling(y_values, -10, 10)
# pos_norm = dict()
# for i in range(len(x_values)):
#     pos_norm[getDictKeys(pos)[i]] = (x_values_norm[i], y_values_norm[i])
# pos = pos_norm
# nodes = t.unique_points
# trajectories = t.trajectories
# edges = t.constructWeightMatrix(trajectories)
# edges = (edges / np.max(edges)) * 5




# draw_weighted_graph(nodes, edges, pos)