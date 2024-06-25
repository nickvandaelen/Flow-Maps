import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
import time
import copy
import itertools
import math


def is_point_on_segment(px, py, x1, y1, x2, y2, atol, rtol):
    """ Check if point (px, py) is on the line segment from (x1, y1) to (x2, y2) with given tolerance. """
    if min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2):
        if np.isclose((px - x1) * (y2 - y1), (x2 - x1) * (py - y1), atol=atol, rtol=rtol):
            return True
    return False

def check_intersection(px, py, x1, y1, x2, y2, node_size, edge_width):
    """ Check if a point intersects a line segment based on node size and edge width. """
    atol = node_size / 100  # Tolerance based on half the node size
    rtol = edge_width / 100  # Tolerance based on half the edge width
    if min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2):
        if np.isclose((px - x1) * (y2 - y1), (x2 - x1) * (py - y1), atol=atol, rtol=rtol):
            return 1.0  # Consider this as an intersection
        elif np.isclose((px - x1) * (y2 - y1), (x2 - x1) * (py - y1), atol=atol * 2, rtol=rtol * 2):
            return 0.5  # Close but not intersecting
        elif np.isclose((px - x1) * (y2 - y1), (x2 - x1) * (py - y1), atol=atol * 5, rtol=rtol * 5):
            return 0.25
    return 0.0  # No intersection

def node_spacing(pos):
    min_dist = 99999
    x_values = [point[0] for point in pos.values()]
    dim = max(x_values) - min(x_values)
    fact = (10/dim)
    for i in range(len(nodes)):
        for j in range(i+1,len(nodes)):

            dist = math.sqrt(fact *((pos[nodes[i]][0] - pos[nodes[j]][0]) ** 2 + (pos[nodes[i]][1] - pos[nodes[j]][1]) ** 2))
            if dist < min_dist:
                min_dist = dist
                
    return min_dist
            

def calculate_score(intersection_count, distance_moved, order_preserved, min_dist, dim, print_results = False):
    int_pen = -(intersection_count*1000)
    dist_pen = ((-1000*distance_moved)/dim)
    order_pen = (-order_preserved*800)
    spacing_pen = (min_dist*100)
    
    combined = int_pen + dist_pen + order_pen + spacing_pen
    if print_results:
        print('int_pen {},{}; dist_pen {},{}; order_pen {}, {}; spacing {}, {}; total {}'.format(intersection_count, int_pen, distance_moved, dist_pen, order_preserved, order_pen, min_dist, spacing_pen, combined))
    
    return combined


def check_order_preservation(pos1, pos2):
    switches = 0
    keys = list(pos1.keys())
        
    # Compare horizontal positions
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            key_i = keys[i]
            key_j = keys[j]
            # Check horizontal order in true_pos
            if (pos1[key_i][0] - pos1[key_j][0]) * (pos2[key_i][0] - pos2[key_j][0]) < 0:
                switches += 1
                
            # Check vertical order in true_pos
            if (pos1[key_i][1] - pos1[key_j][1]) * (pos2[key_i][1] - pos2[key_j][1]) < 0:
                switches += 1
        
    return switches
    

def total_distance_moved(true_pos, plotted_pos):
    def euclidean_distance(point1, point2):
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
    
    total_distance = 0.0
    for key in true_pos:
        total_distance += euclidean_distance(true_pos[key], plotted_pos[key])
    
    return total_distance


pos = {'A': (-3, 1), 'B': (-2, -2), 'C': (0.5, 0), 'D': (0, -1), 'E': (3, 1), 'F': (3.5, -2), 'J':(-0.75, -1.5)}
  
true_pos = pos

plotted_pos = {'A': (-3, 1), 'B': (-4, -3), 'C': (0.5, 0), 'D': (0, -2.5), 'E': (3, 1), 'F': (3.5, -2), 'J':(-0.75, -1.5)}

cnt = check_order_preservation(true_pos, plotted_pos)
print(total_distance_moved(true_pos, plotted_pos))

def draw_weighted_graph(nodes, edges, positions):
    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node)
    pos = positions

    x_values = [point[0] for point in pos.values()]
    plt.xlim(min(x_values) - 1, max(x_values) + 1)

    seg = (max(x_values) - min(x_values))/50
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            weight = edges[i][j]
            node1 = nodes[i]
            node2 = nodes[j]
            x1,y1 = pos[node1]
            x2,y2 = pos[node2]
            rise=(y2-y1)
            run=(x2-x1)-(seg*2)
            if run != 0:
                dir_exit = np.arctan2(rise, (-1*run))
                dir_enter = np.arctan2(-rise, -run) 
                if dir_exit < 0:
                    dir_exit += 2*np.pi
                if dir_enter < 0:
                    dir_enter += 2*np.pi
            elif y1 > y2:
                dir_exit = 3*np.pi/2
                dir_enter = np.pi/2
            else:
                dir_exit = np.pi/2
                dir_enter = 3*np.pi/2
            if weight != 0:
                G.add_edge(nodes[i], nodes[j], weight=weight, d1 = round(dir_exit,3), d2 = round(dir_enter,3))

    for node in G.nodes():
        successor_edges = sorted(G.edges(node, data=True), key=lambda x: x[2]['d1'])
        p1_counter = 0
        p2_counter = 0
        predecessor_edges = sorted(G.in_edges(node, data=True), key=lambda x: x[2]['d2'])
        for u, v, edge_data in successor_edges:
            G.edges[u, v]['p1'] = p1_counter
            p1_counter += 1
        for u, v, edge_data in predecessor_edges:
            G.edges[u, v]['p2'] = p2_counter
            p2_counter += 1

    out_edge_weights = {}
    in_edge_weights = {}
    for node in nodes:
        out_edge_weights[node] = sum(G.edges[edge]['weight'] for edge in G.edges() if edge[0] == node)
        in_edge_weights[node] = sum(G.edges[edge]['weight'] for edge in G.edges() if edge[1] == node)

    cmap = cm.get_cmap('viridis')

    x_values = [point[0] for point in pos.values()]
    plt.xlim(min(x_values) - 1, max(x_values) + 1)
    
    dim = max(x_values) - min(x_values)
    
    fact = 600/(dim)

    running_in_width = {node: 0 for node in nodes}
    running_out_width = {node: 0 for node in nodes}

    p1_counters = {node: 0 for node in nodes}
    p2_counters = {node: 0 for node in nodes}
    edges_placed = 0
    intersection_count = 0
    while edges_placed < len(G.edges()):
        for edge in G.edges():
            node1, node2 = edge

            if G.edges[edge]['p1'] == p1_counters[node1] and G.edges[edge]['p2'] == p2_counters[node2]:

                weight = G.edges[edge]['weight']
                x1, y1 = pos[node1]
                x2, y2 = pos[node2]
                seg = seg

                control_point = (x1 + seg, y1 + out_edge_weights[node1]/fact-running_out_width[node1]/(fact/2)-weight/fact)
                control_point_2 = (x2 - seg, y2 + in_edge_weights[node2]/fact-running_in_width[node2]/(fact/2)-weight/fact)

                color_value = (y1 - min(pos[node][1] for node in nodes)) / (max(pos[node][1] for node in nodes) - min(pos[node][1] for node in nodes))
                color = 'red'#cmap(color_value)
                '''
                plt.plot([x1, control_point[0]], [y1+ out_edge_weights[node1]/fact-running_out_width[node1]/(fact/2)-weight/fact, control_point[1]], color=color, linewidth=weight, alpha=min(1/weight+0.5, 1), solid_capstyle="butt")

                for node in nodes:
                    if node != node1 and node != node2:
                        px, py = pos[node]
                        intersection_count += check_intersection(px, py, x1, y1, control_point[0], control_point[1], weight, out_edge_weights)
                '''
                plt.plot([control_point[0], control_point_2[0]], [control_point[1], control_point_2[1]], color=color, linewidth=weight, alpha=min(1/weight+0.5, 1), solid_capstyle="round")

                for node in nodes:
                    if node != node1 and node != node2:
                        px, py = pos[node]
                        intersection_count += check_intersection(px, py, control_point[0], control_point[1], control_point_2[0], control_point_2[1], weight, out_edge_weights[node])
                '''
                plt.plot([control_point_2[0], x2], [control_point_2[1], y2+ in_edge_weights[node2]/fact-running_in_width[node2]/(fact/2)-weight/fact], color=color, linewidth=weight, alpha=min(1/weight+0.5, 1), solid_capstyle="butt")

                for node in nodes:
                    if node != node1 and node != node2:
                        px, py = pos[node]
                        intersection_count += check_intersection(px, py, control_point_2[0], control_point_2[1], x2, y2, weight, out_edge_weights)
                '''
                running_out_width[node1] += weight
                running_in_width[node2] += weight

                p1_counters[node1] += 1
                p2_counters[node2] += 1
                edges_placed += 1

    print(f"Total intersections: {intersection_count}")

    node_patches = [patches.Circle((0, 0), max(out_edge_weights[node], in_edge_weights[node])/50, edgecolor='black', facecolor='black') for node in nodes]

    for i, patch in enumerate(node_patches):
        x, y = pos[nodes[i]]
        patch.center = (x, y)
        plt.gca().add_patch(patch)
        plt.gca().text(x, y, nodes[i], color='white', ha='center', va='center', fontsize=8)


    plt.axis('equal')
    plt.axis('off')
    plt.show()



def test_weighted_graph(nodes, edges, positions):
    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node)
    pos = positions

    x_values = [point[0] for point in pos.values()]
    plt.xlim(min(x_values) - 1, max(x_values) + 1)

    seg = (max(x_values) - min(x_values))/50
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            weight = edges[i][j]
            node1 = nodes[i]
            node2 = nodes[j]
            x1,y1 = pos[node1]
            x2,y2 = pos[node2]
            rise=(y2-y1)
            run=(x2-x1)-(seg*2)
            if run != 0:
                dir_exit = np.arctan2(rise, (-1*run))
                dir_enter = np.arctan2(-rise, -run) 
                if dir_exit < 0:
                    dir_exit += 2*np.pi
                if dir_enter < 0:
                    dir_enter += 2*np.pi
            elif y1 > y2:
                dir_exit = 3*np.pi/2
                dir_enter = np.pi/2
            else:
                dir_exit = np.pi/2
                dir_enter = 3*np.pi/2
            if weight != 0:
                G.add_edge(nodes[i], nodes[j], weight=weight, d1 = round(dir_exit,3), d2 = round(dir_enter,3))

    for node in G.nodes():
        successor_edges = sorted(G.edges(node, data=True), key=lambda x: x[2]['d1'])
        p1_counter = 0
        p2_counter = 0
        predecessor_edges = sorted(G.in_edges(node, data=True), key=lambda x: x[2]['d2'])
        for u, v, edge_data in successor_edges:
            G.edges[u, v]['p1'] = p1_counter
            p1_counter += 1
        for u, v, edge_data in predecessor_edges:
            G.edges[u, v]['p2'] = p2_counter
            p2_counter += 1

    out_edge_weights = {}
    in_edge_weights = {}
    for node in nodes:
        out_edge_weights[node] = sum(G.edges[edge]['weight'] for edge in G.edges() if edge[0] == node)
        in_edge_weights[node] = sum(G.edges[edge]['weight'] for edge in G.edges() if edge[1] == node)

    cmap = cm.get_cmap('viridis')

    x_values = [point[0] for point in pos.values()]
    plt.xlim(min(x_values) - 1, max(x_values) + 1)
    
    dim = max(x_values) - min(x_values)
    
    fact = 600/(dim)

    running_in_width = {node: 0 for node in nodes}
    running_out_width = {node: 0 for node in nodes}

    p1_counters = {node: 0 for node in nodes}
    p2_counters = {node: 0 for node in nodes}
    edges_placed = 0
    intersection_count = 0
    while edges_placed < len(G.edges()):
        for edge in G.edges():
            node1, node2 = edge

            if G.edges[edge]['p1'] == p1_counters[node1] and G.edges[edge]['p2'] == p2_counters[node2]:

                weight = G.edges[edge]['weight']
                x1, y1 = pos[node1]
                x2, y2 = pos[node2]
                seg = seg

                control_point = (x1 + seg, y1 + out_edge_weights[node1]/fact-running_out_width[node1]/(fact/2)-weight/fact)
                control_point_2 = (x2 - seg, y2 + in_edge_weights[node2]/fact-running_in_width[node2]/(fact/2)-weight/fact)

                color_value = (y1 - min(pos[node][1] for node in nodes)) / (max(pos[node][1] for node in nodes) - min(pos[node][1] for node in nodes))
                color = cmap(color_value)
                '''
                plt.plot([x1, control_point[0]], [y1+ out_edge_weights[node1]/fact-running_out_width[node1]/(fact/2)-weight/fact, control_point[1]], color=color, linewidth=weight, alpha=min(1/weight+0.5, 1), solid_capstyle="butt")

                for node in nodes:
                    if node != node1 and node != node2:
                        px, py = pos[node]
                        intersection_count += check_intersection(px, py, x1, y1, control_point[0], control_point[1], weight, out_edge_weights)
                '''
                #plt.plot([control_point[0], control_point_2[0]], [control_point[1], control_point_2[1]], color=color, linewidth=weight, alpha=min(1/weight+0.5, 1), solid_capstyle="round")

                for node in nodes:
                    if node != node1 and node != node2:
                        px, py = pos[node]
                        intersection_count += check_intersection(px, py, control_point[0], control_point[1], control_point_2[0], control_point_2[1], weight, out_edge_weights[node])
                '''
                plt.plot([control_point_2[0], x2], [control_point_2[1], y2+ in_edge_weights[node2]/fact-running_in_width[node2]/(fact/2)-weight/fact], color=color, linewidth=weight, alpha=min(1/weight+0.5, 1), solid_capstyle="butt")

                for node in nodes:
                    if node != node1 and node != node2:
                        px, py = pos[node]
                        intersection_count += check_intersection(px, py, control_point_2[0], control_point_2[1], x2, y2, weight, out_edge_weights)
                '''
                running_out_width[node1] += weight
                running_in_width[node2] += weight

                p1_counters[node1] += 1
                p2_counters[node2] += 1
                edges_placed += 1
    return intersection_count



np.arange(-1, 1.001, 0.5)


def move_one_algorithm(nodes, edge, pos, max_move, step):
    max_score = -999999
    count = 0
    best_positions = copy.deepcopy(pos)
    x_values = [point[0] for point in pos.values()]
    
    moves = np.arange(-max_move, (max_move+.001), step)
    
    dim = max(x_values) - min(x_values)
    for node in nodes:
        for x in moves:
            for y in moves:
                count = count+1
                new_pos = copy.deepcopy(pos)
                
                new_pos[node] = (pos[node][0]+x, pos[node][1]+y)
                
                ints = test_weighted_graph(nodes, edges, new_pos)
                movement = total_distance_moved(pos, new_pos)
                order = check_order_preservation(pos, new_pos)
                min_dist = node_spacing(new_pos)
                score = calculate_score(ints, movement, order, min_dist, dim)
                if score > max_score:
                    max_score = score
                    print(max_score)
                    print(node, x, y)
                    best_positions = copy.deepcopy(new_pos)
                    
    print(best_positions) 
    print(count)
    
    
    draw_weighted_graph(nodes, edges, best_positions)
    print('Order changed: {}'.format(check_order_preservation(pos, best_positions)))
    print('Distance Moved: {}'.format(total_distance_moved(pos, best_positions)))
    print('closest nodes {}'.format(node_spacing(best_positions)))

    
def brute_force_algorithm(nodes, edges, pos):
    best_positions = copy.deepcopy(pos)
    best_score = -float('inf')
    
    x_values = [point[0] for point in pos.values()]
    dim = max(x_values) - min(x_values)
    
    movements = [1,0,1]
    all_combinations = list(itertools.product(movements, repeat=len(nodes)*2))  # Generate all combinations of movements
    c = 0
    for moves in all_combinations:
        new_pos = copy.deepcopy(pos)
        for i, node in enumerate(nodes):
            new_pos[node] = (pos[node][0] + moves[2*i], pos[node][1] + moves[2*i+1])
        
        # Calculate metrics
        ints = test_weighted_graph(nodes, edges, new_pos)  # Example function to compute some metric
        movement = total_distance_moved(pos, new_pos)
        order = check_order_preservation(pos, new_pos)
        
        # Calculate score based on metrics
        score = calculate_score(ints, movement, order, dim)
        
        # Update best positions and score if this configuration is better
        if score > best_score:
            best_score = score
            best_positions = copy.deepcopy(new_pos)
    
    print("Best positions:", best_positions)
    print("Best score:", best_score)
    
    draw_weighted_graph(nodes, edges, best_positions)
    print(check_order_preservation(pos, best_positions))
    print(total_distance_moved(pos, best_positions))
    
    
import random
    

    
    
def iterative_algorithm(nodes, edges, pos, max_move, step, iterations=10):
    best_positions = copy.deepcopy(pos)
    best_score = -float('inf')
    
    moves = np.arange(-max_move, (max_move+.001), step)
    
    x_values = [point[0] for point in pos.values()]
    dim = max(x_values) - min(x_values)
    
    for it in range(iterations):
        improved = False
        for node in random.sample(nodes, len(nodes)):
            local_best_score = -float('inf')
            local_best_pos = best_positions[node]
            
            for x in moves:
                for y in moves:
                    if x == 0 and y == 0:
                        continue
                    new_pos = copy.deepcopy(best_positions)
                    new_pos[node] = (best_positions[node][0] + x, best_positions[node][1] + y)
                    
                    # Calculate metrics
                    ints = test_weighted_graph(nodes, edges, new_pos)
                    movement = total_distance_moved(pos, new_pos)
                    order = check_order_preservation(pos, new_pos)
                    min_dist = node_spacing(new_pos)
                    # Calculate score based on metrics
                    score = calculate_score(ints, movement, order, min_dist, dim)
                    
                    if score > local_best_score:
                        local_best_score = score
                        local_best_pos = new_pos[node]
            
            # Update the position if it improved
            if local_best_score > best_score:
                best_score = local_best_score
                best_positions[node] = local_best_pos
                improved = True
                
        ints = test_weighted_graph(nodes, edges, best_positions)
        movement = total_distance_moved(pos, best_positions)
        order = check_order_preservation(pos, best_positions)
        min_dist = node_spacing(best_positions)
        
        score = calculate_score(ints, movement, order, min_dist, dim, True)
        print('#############iter {}, score {}#############'.format(it, score))
        if not improved:
            break
    
    print("Best positions after hybrid algorithm:", best_positions)
    print("Best score:", best_score)
    
    draw_weighted_graph(nodes, edges, best_positions)
    print('Order changed: {}'.format(check_order_preservation(pos, best_positions)))
    print('Distance Moved: {}'.format(total_distance_moved(pos, best_positions)))
    print('closest nodes {}'.format(node_spacing(best_positions)))



start_time = time.time()

nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
pos = {'A': (-6, 1), 'B': (-5, -1), 'C': (-4, 2), 'D': (-3, -2), 'E': (-2, 0), 'F': (-1, -1), 
       'G': (0, 1), 'H': (1, -2), 'I': (2, 0), 'J': (3, 2), 'K': (4, -1), 'L': (5, 1)}

# Define the adjacency matrix for the edges
edges = np.array([
    [0, 5, 2, 5, 0, 0, 0, 3, 0, 0, 0, 4], 
    [0, 0, 1, 2, 1, 0, 0, 0, 0, 5, 0, 1], 
    [0, 0, 0, 0, 8, 3, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 2, 4, 0, 0, 1, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 7, 2, 0, 0, 2, 0], 
    [0, 0, 0, 0, 0, 0, 4, 5, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 3, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 4], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 2], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 3], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

# Example usage:
nodes_small = ['A', 'B', 'C', 'D', 'E', 'F', 'J']
edges_small = np.array([[0, 5, 2, 5, 0, 0, 0], 
                  [3, 0, 1, 2, 1, 0, 5], 
                  [0, 0, 0, 0, 8, 3, 0], 
                  [0, 0, 0, 0, 2, 4, 0], 
                  [0, 0, 0, 0, 0, 0, 0], 
                  [0, 0, 0, 0, 0, 0, 0], 
                  [0, 0, 0, 0, 3, 2, 0]])
pos_small = {'A': (-3, 1), 'B': (-2, -2), 'C': (0.5, 0), 'D': (0, -1), 'E': (3, 1), 'F': (3.5, -2), 'J':(-0.75, -1.5)}


draw_weighted_graph(nodes, edges, pos)
               
#move_one_algorithm(nodes, edges, pos, 1, 1)                
                
           
                
           
            
start_time = time.time()
 
iterative_algorithm(nodes, edges, pos, 0.5, 0.1)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds") 

# Example usage:
nodes_smaller = ['A', 'B', 'C', 'D']
edges_smaller = np.array([[0, 5, 2, 5], 
                  [3, 0, 1,3], 
                  [0, 0, 0,3],
                  [0,0,0,0]])
pos_smaller = {'A': (-3, 1), 'B': (-2, -2), 'C': (0.5, 0), 'D':(-0.8,-1)}

