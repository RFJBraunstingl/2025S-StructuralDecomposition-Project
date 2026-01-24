
path_to_input = 'samples/C5.gr'
path_to_input = 'samples/balaban_10cage.gr'
path_to_input = 'samples/star7.gr'
path_to_input = 'samples/two-levels.gr'

class Graph:
    adjacency_list = {}

    @staticmethod
    def create_for_num_of_nodes(num_of_nodes):
        adjacency_list = {}
        for i in range(num_of_nodes):
            adjacency_list[i + 1] = []

        return Graph(adjacency_list)

    def __init__(self, adjacency_list: dict):
        self.adjacency_list = adjacency_list

    def create_edge(self, u, v):
        self.adjacency_list[u].append(v)
        self.adjacency_list[v].append(u)

    def copy(self):
        copy_of_adjacency_list = {}
        for x in self.adjacency_list:
            copy_of_adjacency_list[x] = self.adjacency_list[x].copy()

        return Graph(copy_of_adjacency_list)

    def get_node_with_min_degree(self):
        node = None
        min_degree = 0
        for v in self.adjacency_list:
            if node is None:
                node = v
                min_degree = len(self.adjacency_list[v])
                continue

            if min_degree > len(self.adjacency_list[v]):
                min_degree = len(self.adjacency_list[v])
                node = v

        return node

    def get_num_of_nodes(self):
        return len(self.adjacency_list)

    def remove_node_and_fill_edges(self, node: int):
        neighbors = self.adjacency_list[node]
        for i in range(len(neighbors)):
            neighbour = neighbors[i]
            # remove node to be popped from neighbour list
            self.adjacency_list[neighbour].remove(node)
            # connect all following neighbours to this one
            j = i + 1
            while j < len(neighbors):
                other_neighbour = neighbors[j]
                if not self.is_connected(neighbour, other_neighbour):
                    self.create_edge(neighbour, other_neighbour)
                j += 1

        self.adjacency_list.pop(node)

    def is_connected(self, u, v):
        return v in self.adjacency_list[u]

    def get_neighbours(self, node):
        return self.adjacency_list[node]

    def __str__(self):
        return str(self.adjacency_list)


print("parsing input file...")
with open(path_to_input, 'r') as in_file:
    for line in in_file:
        line = line.strip()
        if line.startswith('c'):
            continue

        parts = line.split(" ")
        if line.startswith('p'):
            if len(parts) < 4:
                raise RuntimeError("p line should have 4 parts")

            num_of_nodes = parts[2]
            node_set = frozenset([i for i in range(int(num_of_nodes))])
            graph = Graph.create_for_num_of_nodes(int(num_of_nodes))
            continue

        if not graph:
            raise RuntimeError("p line should have come first")

        u = int(parts[0])
        v = int(parts[1])
        graph.create_edge(u, v)


print("parsed input file with {} nodes".format(len(node_set)))
print("parsed graph...", graph)

print("building tree decomposition using min degree heuristic")
# create an elimination ordering and construct the bags
g = graph.copy()
elimination_ordering = []
while g.get_num_of_nodes() > 0:
    node_to_remove = g.get_node_with_min_degree()
    list_of_nodes_in_bag = [node_to_remove]
    list_of_nodes_in_bag += g.get_neighbours(node_to_remove)
    bag = frozenset(list_of_nodes_in_bag)

    elimination_ordering.append((node_to_remove, bag))
    g.remove_node_and_fill_edges(node_to_remove)

    # print("elimination ordering...", elimination_ordering)
    # print("graph after removing...", g)

print("final elimination ordering...", elimination_ordering)


def get_index_of(element, list):
    for i in range(len(list)):
        if list[i] == element:
            return i

    raise RuntimeError("element not in list")


print("building tree decomposition from elimination ordering...")
all_nodes = [x[0] for x in elimination_ordering]
all_bags = [x[1] for x in elimination_ordering]
adjacency_list = {}
for bag in all_bags:
    adjacency_list[bag] = []

tree_decomposition_graph = Graph(adjacency_list)

for element in elimination_ordering:
    graph_node: int = element[0]
    graph_bag: frozenset = element[1]

    # get the neighbour with the lowest index and connect the bag to that node
    lowest_index = None
    lowest_index_neighbour = None
    for neighbour_node in graph_bag:
        if neighbour_node == graph_node:
            continue

        index = get_index_of(neighbour_node, all_nodes)
        neighbour_bag = all_bags[index]
        if lowest_index is None:
            lowest_index = index
            lowest_index_neighbour = neighbour_bag
            continue

        if index < lowest_index:
            lowest_index = index
            lowest_index_neighbour = neighbour_bag

    if lowest_index_neighbour is not None:
        tree_decomposition_graph.create_edge(graph_bag, lowest_index_neighbour)

print("constructed tree decomposition...", tree_decomposition_graph)

print("transforming tree composition into nice one...")
