import networkx as nx
from networkx.algorithms import approximation
from nice_tree_decomposition import make_nice_tree_decomposition

path_to_gr_file = 'samples/two-levels.gr'


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
make_nice_tree_decomposition(decomposition)
print("nice nodes", decomposition.nodes)
print("nice edges", decomposition.edges)

if len(decomposition.nodes) < 1:
    raise RuntimeError("empty decomposition encountered")

# *** begin algorithm ***
# start at the leafs and calculate best solutions
visited_nodes_to_statistics_set = {}


# produce a map which holds each node (as key) and the number of unvisited neighbours (as value)
def map_nodes_to_neighbours(nodes, edges):
    unvisited_neighbours_map = {}
    visited_neighbours_map = {}
    for node in nodes:
        visited_neighbours_map[node] = []
        unvisited_neighbours_map[node] = []

    for u, v in edges:
        if u in visited_nodes_to_statistics_set:
            visited_neighbours_map[v].append(u)
        else:
            unvisited_neighbours_map[v].append(u)

        if v in visited_nodes_to_statistics_set:
            visited_neighbours_map[u].append(v)
        else:
            unvisited_neighbours_map[u].append(v)

    return unvisited_neighbours_map, visited_neighbours_map


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
            for vertex in permutation:
                selected_set.add(vertex)

            current_length = len(selected_set)
            if min_selected_set_length == 0 or current_length < min_selected_set_length:
                min_selected_set = selected_set
                min_selected_set_length = current_length

        visited_nodes_to_statistics_set[node][selected_vertex] = min_selected_set


# NOTE: networkx gives us root to leafs from left to right
# => thus we can traverse the list in reverse order to go from leafs to root
nodelist = []
for v in decomposition.nodes:
    nodelist.append(v)

root_node = nodelist[0]
nodelist = reversed(nodelist)
for node in nodelist:
    unvisited_neighbours, visited_neighbours = map_nodes_to_neighbours(decomposition.nodes, decomposition.edges)
    visit_node(node, visited_neighbours[node])


print("root node", root_node)
min_solution = None
root_node_statistics = visited_nodes_to_statistics_set[root_node]
for root_node_vertex in root_node_statistics:
    solution = root_node_statistics[root_node_vertex]
    if min_solution is None or len(solution) < len(min_solution):
        min_solution = solution

print("min solution", min_solution)