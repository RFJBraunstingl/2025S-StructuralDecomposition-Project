import math

path_to_input = 'samples/C5.gr'
path_to_input = 'samples/two-levels.gr'
path_to_input = 'samples/star7.gr'


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

    def for_each_node_with_neighbours(self, consumer):
        for node in self.adjacency_list:
            consumer(node, adjacency_list[node])

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

    def get_node_with_max_degree(self):
        node = None
        max_degree = 0
        for v in self.adjacency_list:
            if node is None:
                node = v
                max_degree = len(self.adjacency_list[v])
                continue

            if max_degree < len(self.adjacency_list[v]):
                max_degree = len(self.adjacency_list[v])
                node = v

        return node

    def get_num_of_nodes(self):
        return len(self.adjacency_list)

    def remove_node(self, node):
        neighbors = self.adjacency_list[node]
        for i in range(len(neighbors)):
            neighbour = neighbors[i]
            # remove node to be popped from neighbour list
            self.adjacency_list[neighbour].remove(node)

        self.adjacency_list.pop(node)

    def remove_node_and_fill_edges(self, node):
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
# node types
ROOT_NODE = "ROOT"
JOIN_NODE = "JOIN"
INTRODUCE_NODE = "INTRODUCE"
FORGET_NODE = "FORGET"
LEAF_NODE = "LEAF"
INTRODUCE_EDGE_NODE = "INTRODUCE_EDGE"


class NiceTreeDecompositionNode:
    node_type: str = ""
    child_nodes = []
    bag = frozenset()

    def __str__(self):
        meta = " "
        if self.node_type == INTRODUCE_EDGE_NODE:
            meta = f"({self.introduced_edge})"
        else:
            meta = f"{str(self.bag)}"
        return f"{self.node_type} {meta}: [{', '.join([str(n) for n in self.child_nodes])}]"

    def add_children(self, children, tree_decomposition_graph: Graph):
        if len(children) == 0:
            # add leaf node
            leaf_node = NiceTreeDecompositionNode()
            leaf_node.node_type = LEAF_NODE
            self.child_nodes = [leaf_node]
            return

        # if we add more than 1 child, this node becomes a join node
        if len(children) > 1:
            half_index = len(children) // 2

            child_1 = NiceTreeDecompositionNode()
            child_1.bag = self.bag
            child_1.add_children(children[:half_index], tree_decomposition_graph)
            child_2 = NiceTreeDecompositionNode()
            child_2.bag = self.bag
            child_2.add_children(children[half_index:], tree_decomposition_graph)

            self.node_type = JOIN_NODE
            self.child_nodes = [child_1, child_2]

        else:
            bag = children[0]
            neighbours = tree_decomposition_graph.get_neighbours(bag)
            tree_decomposition_graph.remove_node(bag)

            new_child = NiceTreeDecompositionNode()
            new_child.bag = bag
            new_child.add_children(neighbours, tree_decomposition_graph)
            self.child_nodes = [new_child]

    def ensure_bag_diff_one(self):
        for child in self.child_nodes:
            child.ensure_bag_diff_one()

            if child.bag == self.bag:
                continue

            # print("ensure bag diff from ", child.bag, " to ", self.bag)
            # generate a sequence of bags with diff 1
            to_be_forgotten = child.bag.difference(self.bag)
            to_be_introduced = self.bag.difference(child.bag)
            sequence = []
            current_bag = set(child.bag)
            for f in to_be_forgotten:
                current_bag.remove(f)
                forget_node = NiceTreeDecompositionNode()
                forget_node.bag = frozenset(current_bag)
                forget_node.forgotten_vertex = f
                sequence.append(forget_node)
            for i in to_be_introduced:
                current_bag.add(i)
                introduce_node = NiceTreeDecompositionNode()
                introduce_node.bag = frozenset(current_bag)
                introduce_node.introduced_vertex = i
                sequence.append(introduce_node)

            # we don't need the last entry since it must be equivalent to self.bag
            sequence = sequence[:-1]
            # print("sequence:", ", ".join([str(x) for x in sequence]))

            if len(sequence) > 0:
                current_child = child
                for node in sequence:
                    node.child_nodes = [current_child]
                    current_child = node

                self.child_nodes.remove(child)
                self.child_nodes.append(current_child)

    def ensure_consistent_node_type(self):
        for child in self.child_nodes:
            child.ensure_consistent_node_type()

        if len(self.child_nodes) > 2:
            raise RuntimeError("nodes must have exactly 1 or exactly 2 children")

        if len(self.child_nodes) == 2:
            if self.bag != self.child_nodes[0].bag or self.bag != self.child_nodes[1].bag:
                raise RuntimeError("nodes with 2 children must have equivalent bags")

            # nodes with 2 children are join nodes
            self.node_type = JOIN_NODE

        elif len(self.child_nodes) == 1:
            # nodes with 1 child which have more entries in the bag are introduce nodes
            if len(self.bag) > len(self.child_nodes[0].bag):
                self.node_type = INTRODUCE_NODE
            # nodes with 1 child which have less entries in the bag are forget nodes
            elif len(self.bag) < len(self.child_nodes[0].bag):
                self.node_type = FORGET_NODE

    def insert_before_child(self, child, node_to_introduce):
        if child in self.child_nodes:
            self.child_nodes.remove(child)

        node_to_introduce.child_nodes.append(child)
        self.child_nodes.append(node_to_introduce)

    @staticmethod
    def get_edges_covered_by_bag(edge_collection, bag):
        result = []
        for edge in edge_collection:
            if edge.issubset(bag):
                result.append(edge)

        return result

    def add_introduce_edge_nodes(self, edge_collection):
        for edge in NiceTreeDecompositionNode.get_edges_covered_by_bag(edge_collection, self.bag):
            # transform this node to introduce edge and add a copy as child
            replacement_node = NiceTreeDecompositionNode()
            replacement_node.node_type = self.node_type
            replacement_node.bag = self.bag
            replacement_node.child_nodes = self.child_nodes
            if hasattr(self, 'introduced_edge'):
                replacement_node.introduced_edge = self.introduced_edge

            self.child_nodes = [replacement_node]
            self.node_type = INTRODUCE_EDGE_NODE
            self.introduced_edge = edge
            edge_collection.remove(edge)

        for child in self.child_nodes:
            child.add_introduce_edge_nodes(edge_collection)


class NiceTreeDecomposition:
    root_node: NiceTreeDecompositionNode

    @staticmethod
    def create_from_tree_decomposition(tree_decomposition: Graph):
        # input is a graph where every node is a bag
        print("choose an arbitrary root node...")
        root_node = NiceTreeDecompositionNode()
        root_node.bag = frozenset()  # root has empty bag
        root_node.node_type = ROOT_NODE

        print("starting from root, insert all neighbours as children ensuring that the tree is binary...")
        root_node.add_children([tree_decomposition_graph.get_node_with_max_degree()], tree_decomposition)

        print("inserting introduce/forget nodes...")
        root_node.ensure_bag_diff_one()

        print("cleanup node types")
        root_node.ensure_consistent_node_type()

        return root_node


nice_tree_decomposition = NiceTreeDecomposition.create_from_tree_decomposition(tree_decomposition_graph)
print(nice_tree_decomposition)

print("collecting edges for graph " + str(graph.adjacency_list))
edges = set()
adjacency_list = graph.adjacency_list
for u in adjacency_list:
    for v in adjacency_list[u]:
        edges.add(frozenset([u,v]))

print("collected edges: ", ", ".join([str(x) for x in edges]))
print("inserting introduce edge nodes")
nice_tree_decomposition.add_introduce_edge_nodes(edges)
print(nice_tree_decomposition)

# vertex colors
BLACK = 0
WHITE = 1
GREY = 2


# generate all 3^|X_t| colorings of a bag X_t
# we encode colorings as strings like "1:0,2:2,3:1"
# ...where 1,2,3 are sorted vertices and
# ...and 0,2,1 are the colors of the nodes
def generate_colorings_recursively(l):
    if len(l) == 0:
        yield []
        return

    for recurse in generate_colorings_recursively(l[1:]):
        yield [f"{l[0]}:0"] + recurse
        yield [f"{l[0]}:1"] + recurse
        yield [f"{l[0]}:2"] + recurse


def generate_colorings(bag):
    # a bag here is a set of vertices
    if len(bag) == 0:
        return []

    vertices = sorted([x for x in bag])
    for coloring in generate_colorings_recursively(vertices):
        yield ",".join(coloring)


def coloring_without_vertex(coloring, v):
    return ",".join([x for x in coloring.split(",") if not x.startswith(f"{v}:")])


def vertex_has_color(coloring, v, color):
    return f"{v}:{color}" in coloring.split(",")


def coloring_with_replaced_vertex_color(coloring, v, new_color):
    filtered_list = [x for x in coloring.split(",") if not x.startswith(f"{v}:")]
    replacement = [f"{v}:{new_color}"]
    new_list = filtered_list + replacement
    new_list.sort(key=lambda x: int(x.split(':')[0]))
    return ",".join(new_list)


def is_consistent(f1, f2, f):
    f_pairs = [x.split(":") for x in f.split(",")]

    f1_dict = {}
    for v, c in [x.split(":") for x in f1.split(",")]:
        f1_dict[v] = int(c)

    f2_dict = {}
    for v, c in [x.split(":") for x in f2.split(",")]:
        f2_dict[v] = int(c)

    for v, c in f_pairs:
        c = int(c)
        if c == BLACK:
            if f1_dict[v] != BLACK or f2_dict[v] != BLACK:
                return False
        elif c == WHITE:
            if not ((f1_dict[v] == GREY and f2_dict[v] == WHITE) or (f1_dict[v] == WHITE and f2_dict[v] == GREY)):
                return False
        else:
            if f1_dict[v] != GREY or f2_dict[v] != GREY:
                return False

    return True


def get_consistent_colorings(coloring, all_f1, all_f2):
    for f1 in all_f1:
        for f2 in all_f2:
            if is_consistent(coloring, f1, f2):
                yield f1, all_f1[f1], f2, all_f2[f2]


def count_vertices_of_color(coloring, color):
    return len([x for x in coloring.split(",") if x.endswith(f":{color}")])


print("start algorithm")


def get_colorings_for_node(node):
    if node.node_type == LEAF_NODE:
        yield "", 0
        return

    child_colorings = {}
    for x in get_colorings_for_node(node.child_nodes[0]):
        child_colorings[x[0]] = x[1]

    if node.node_type == INTRODUCE_NODE:
        introduced_v = [x for x in node.bag.difference(node.child_nodes[0].bag)][0]
        for coloring in generate_colorings(node.bag):
            if vertex_has_color(coloring, introduced_v, WHITE):
                yield coloring, math.inf
            elif vertex_has_color(coloring, introduced_v, GREY):
                yield coloring, child_colorings[coloring_without_vertex(coloring, introduced_v)]
            else:  # black
                yield coloring, 1 + child_colorings[coloring_without_vertex(coloring, introduced_v)]

    elif node.node_type == INTRODUCE_EDGE_NODE:
        introduced_edge = [x for x in node.introduced_edge]
        u = introduced_edge[0]
        v = introduced_edge[1]
        for coloring in generate_colorings(node.bag):
            if vertex_has_color(coloring, u, BLACK) and vertex_has_color(coloring, v, WHITE):
                transformed_coloring = coloring_with_replaced_vertex_color(coloring, v, GREY)
                yield coloring, child_colorings[transformed_coloring]
            elif vertex_has_color(coloring, u, WHITE) and vertex_has_color(coloring, v, BLACK):
                transformed_coloring = coloring_with_replaced_vertex_color(coloring, u, GREY)
                yield coloring, child_colorings[transformed_coloring]
            else:
                yield coloring, child_colorings[coloring]

    elif node.node_type == FORGET_NODE:
        w = [x for x in node.child_nodes[0].bag.difference(node.bag)][0]
        for coloring in generate_colorings(node.bag):
            coloring_black_w = coloring_with_replaced_vertex_color(coloring, w, BLACK)
            c_for_black_w = child_colorings[coloring_black_w]
            coloring_white_w = coloring_with_replaced_vertex_color(coloring, w, WHITE)
            c_for_white_w = child_colorings[coloring_white_w]
            minimum = min(c_for_black_w, c_for_white_w)
            yield coloring, minimum

    elif node.node_type == JOIN_NODE:
        all_f1 = child_colorings
        all_f2 = {}
        for x in get_colorings_for_node(node.child_nodes[1]):
            all_f2[x[0]] = x[1]

        for coloring in generate_colorings(node.bag):
            minimum = math.inf
            for f1, c1, f2, c2 in get_consistent_colorings(coloring, all_f1, all_f2):
                value = c1 + c2 - count_vertices_of_color(coloring, BLACK)
                if value < minimum:
                    minimum = value

            yield coloring, minimum

    else:
        raise RuntimeError("unhandled node type " + str(node.node_type))


sizes = [x[1] for x in get_colorings_for_node(nice_tree_decomposition.child_nodes[0])]
print("min dominating set size is: " + str(min(sizes)))