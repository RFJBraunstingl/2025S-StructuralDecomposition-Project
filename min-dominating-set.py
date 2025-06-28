from itertools import permutations

import networkx as nx
from networkx.algorithms import approximation

path_to_gr_file = 'samples/balaban_10cage.gr'


def load_gr_file(filepath):
    G = nx.Graph()
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('c'):
                continue  # comment
            elif line.startswith('p'):
                parts = line.strip().split()
                number_of_vertices = int(parts[2])
                number_of_edges = int(parts[3])
                G.add_nodes_from(range(1, number_of_vertices + 1))
            else:
                u, v = line.strip().split()
                G.add_edge(int(u), int(v))
    return G


G = load_gr_file(path_to_gr_file)
print("input parsed...", G)

# get tree decomposition
tree_width, decomposition = approximation.treewidth_min_fill_in(G)
print("found tree decomp with width", tree_width)
print("nodes", decomposition.nodes)
print("edges", decomposition.edges)

# *** begin algorithm ***
# start at the leafs and calculate best solutions
visited_nodes_to_statistics_set = {}


# produce a map which holds each node (as key) and the number of unvisited neighbours (as value)
def map_nodes_to_unvisited_neighbours(nodes, edges):
    result_map = {}
    for node in nodes:
        result_map[node] = []

    for u, v in edges:
        if u not in visited_nodes_to_statistics_set:
            result_map[v].append(u)
        if v not in visited_nodes_to_statistics_set:
            result_map[u].append(v)

    return result_map


def permutations(list_of_lists):
    if len(list_of_lists) == 0:
        return

    if len(list_of_lists) == 1:
        for e in list_of_lists[0]:
            yield [e]

        return

    l = list_of_lists[0]
    for e in l:
        for g in permutations(list_of_lists[1:]):
            yield [e] + g


# calculate the possible solutions taking into account the visited neighbours given
def visit_node(node, list_of_visited_neighbours):
    visited_nodes_to_statistics_set[node] = {}
    for selected_vertex in node:  # nodes are sets of vertices (bags)
        min_selected_set = set()
        min_selected_set.add(selected_vertex)
        min_selected_set_length = 0
        for permutation in permutations(list_of_visited_neighbours):
            selected_set = set()
            selected_set.add(selected_vertex)
            for selected_vertices in permutation:
                selected_set += selected_vertices

            current_length = len(selected_set)
            if min_selected_set_length == 0 or current_length < min_selected_set_length:
                min_selected_set = selected_set
                min_selected_set_length = current_length

        visited_nodes_to_statistics_set[node][selected_vertex] = min_selected_set



# in the beginning, the leafs are those nodes which have count = 1 although no node has been visited yet
node_to_unvisited_neighbour_list_map = map_nodes_to_unvisited_neighbours(decomposition.nodes, decomposition.edges)
for node, neighbour_list in node_to_unvisited_neighbour_list_map.items():
    if len(neighbour_list) == 1:
        # node is a leaf
        visit_node(node, [])

# after the leafs are visited, there has to be always some node which has zero unvisited neighbours
last_node_processed = None
while len(visited_nodes_to_statistics_set) < len(decomposition.nodes):
    unvisited_neighbours = map_nodes_to_unvisited_neighbours(decomposition.nodes, decomposition.edges)
    for node, unvisited_neighbours_set in unvisited_neighbours.items():
        if node in visited_nodes_to_statistics_set:
            continue  # ignore nodes already visited

        if len(unvisited_neighbours_set) > 0:
            continue

        last_node_processed = node