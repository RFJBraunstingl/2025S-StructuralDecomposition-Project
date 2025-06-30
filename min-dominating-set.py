import networkx as nx
from networkx.algorithms import approximation

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
number_of_nodes = len(G.nodes)
adjacency_matrix = []
for i in range(number_of_nodes):
    adjacency_matrix.append([])
    for j in range(number_of_nodes):
        adjacency_matrix[i].append(False)

node_to_adjacency_index = {}
counter = 0
for node in G.nodes:
    node_to_adjacency_index[node] = counter
    counter += 1

for u, v in G.edges:
    index_u = node_to_adjacency_index[u]
    index_v = node_to_adjacency_index[v]

    adjacency_matrix[index_u][index_v] = True
    adjacency_matrix[index_v][index_u] = True


# get tree decomposition
tree_width, decomposition = approximation.treewidth_min_fill_in(G)
print("found tree decomp with width", tree_width)
print("nodes", decomposition.nodes)
print("edges", decomposition.edges)

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


def has_edge(u, v):
    index_u = node_to_adjacency_index[u]
    index_v = node_to_adjacency_index[v]
    return adjacency_matrix[index_u][index_v] == True


def has_neighbour_in(v, possible_solution):
    for other in possible_solution:
        if has_edge(v, other):
            return True

    return False


# a solution is valid if every node is either in the possible solution or is dominated (has a neighbour in) it
def is_valid_solution(possible_solution, node):
    for vertex in node:
        if vertex in possible_solution:
            continue

        if not has_neighbour_in(vertex, possible_solution):
            return False

    return True


def powerset(collection):
    current_element = collection[0]
    yield [current_element]
    if len(collection) == 1:
        return []

    for subset in powerset(collection[1:]):
        yield [current_element] + subset
        yield subset


def visit_node(node, list_of_visited_neighbours):
    visited_nodes_to_statistics_set[node] = {}  # map of lengths to solutions

    # map list of visited neighbours to their minimal solutions
    minimal_solutions_per_neighbour = []
    for neighbour in list_of_visited_neighbours:
        solution_len_dict_for_neighbour = visited_nodes_to_statistics_set[neighbour]
        minimal_solution_length = min(solution_len_dict_for_neighbour.keys())
        list_of_minimal_solutions_for_neighbour = solution_len_dict_for_neighbour[minimal_solution_length]
        minimal_solutions_per_neighbour.append(list_of_minimal_solutions_for_neighbour)

    for possible_solution in powerset([*node]):  # all possible solutions are contained in powerset
        if not is_valid_solution(possible_solution, node):
            continue

        if len(minimal_solutions_per_neighbour) > 0:
            for permutation in permutations(minimal_solutions_per_neighbour):
                combined_solution = set()
                for v in possible_solution:
                    combined_solution.add(v)

                for neighbour_solution in permutation:
                    for v in neighbour_solution:
                        combined_solution.add(v)

                solution_length = len(combined_solution)
                if solution_length not in visited_nodes_to_statistics_set[node]:
                    visited_nodes_to_statistics_set[node][solution_length] = []

                visited_nodes_to_statistics_set[node][solution_length].append(combined_solution)

        else:
            solution_length = len(possible_solution)
            if solution_length not in visited_nodes_to_statistics_set[node]:
                visited_nodes_to_statistics_set[node][solution_length] = []

            visited_nodes_to_statistics_set[node][solution_length].append(possible_solution)


last_node_processed = None
while len(visited_nodes_to_statistics_set) < len(decomposition.nodes):
    unvisited_neighbours, visited_neighbours = map_nodes_to_neighbours(decomposition.nodes, decomposition.edges)
    for node, unvisited_neighbours_set in unvisited_neighbours.items():
        if node in visited_nodes_to_statistics_set:
            continue  # ignore nodes already visited

        if len(unvisited_neighbours_set) > 1:
            continue

        last_node_processed = node
        visit_node(node, visited_neighbours[node])
        break

print("root node", last_node_processed)
root_node_statistics = visited_nodes_to_statistics_set[last_node_processed]
min_solution_length = min(root_node_statistics.keys())
min_solution = root_node_statistics[min_solution_length]

print("min solution:", min_solution)
print("min solution length:", min_solution_length)
