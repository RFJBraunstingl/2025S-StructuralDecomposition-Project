# credit: https://stackoverflow.com/a/79530526
from collections import deque
import networkx


def make_nice_tree_decomposition(T: networkx.Graph):
    """
    Make a tree decomposition nice by ensuring following properties:
    - root node is a empty set
    - every node has at most two children
    - the two children describe the same set of nodes
    - via each edge the bag only adds or forgets one node
    The runtime (experimental) is not worsening the runtime of the program
    """
    leaf_node = __get_leaf_node(T)
    T.add_edge(frozenset(), leaf_node)
    ensure_degree_2(T)
    ensure_children_same_value(T)
    ensure_ignore_add(T)


def ensure_ignore_add(T: networkx.Graph):
    queue = deque([frozenset()])  # Start BFS from node 0
    visited = set()
    count = 0
    while queue:
        count += 1
        b = queue.popleft()

        if b in visited:
            continue
        visited.add(b)

        neighbors = list(T.neighbors(b))

        if len(neighbors) > 2:
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)
        else:
            child = None
            for neighbor in neighbors:
                if neighbor not in visited:
                    child = neighbor
            if child is None:
                # forget all nodes in the bag
                org_b_len = len(b)
                org_b = b
                for i, value_forget in enumerate(b.copy()):
                    if i == org_b_len - 1:
                        break
                    count += 1
                    org_b = frozenset(org_b - frozenset([value_forget]))
                    copy_b = frozenset(org_b | frozenset(["f" + str(count)]))
                    T.add_node(copy_b, original=org_b)
                    T.add_edge(b, copy_b)
                    b = copy_b
            else:
                # parent \ child = {...} those nodes have to be forgotten
                # child \ parent = {...} those nodes have to be add
                # forgotten and add each one step
                T.remove_edge(b, child)
                forget_nodes = (T.nodes[b]["original"] if "original" in T.nodes[b] else b) - (
                    T.nodes[child]["original"] if "original" in T.nodes[child] else child
                )
                add_nodes = (T.nodes[child]["original"] if "original" in T.nodes[child] else child) - (
                    T.nodes[b]["original"] if "original" in T.nodes[b] else b
                )

                org_b = T.nodes[b]["original"] if "original" in T.nodes[b] else b
                for value_forget in forget_nodes:
                    count += 1
                    org_b = frozenset(org_b - frozenset([value_forget]))
                    copy_b = frozenset(org_b | frozenset(["f" + str(count)]))
                    T.add_node(copy_b, original=org_b)
                    T.add_edge(b, copy_b)
                    b = copy_b

                org_b = T.nodes[b]["original"] if "original" in T.nodes[b] else b
                for i, value_add in enumerate(add_nodes):
                    if i == len(add_nodes) - 1:
                        break
                    count += 1
                    org_b = frozenset(org_b | frozenset([value_add]))
                    copy_b = frozenset(org_b | frozenset(["a" + str(count)]))
                    T.add_node(copy_b, original=org_b)
                    T.add_edge(b, copy_b)
                    b = copy_b

                visited.add(b)
                T.add_edge(b, child)
                queue.append(child)


def ensure_children_same_value(T: networkx.Graph):
    queue = deque([frozenset()])
    visited = set()
    count = 0
    while queue:
        count += 1
        b = queue.popleft()

        if b in visited:
            continue
        visited.add(b)

        neighbors = list(T.neighbors(b))

        if len(neighbors) > 2:
            children = []
            for neighbor in neighbors:
                if neighbor not in visited:
                    children.append(neighbor)  # We only have 2 children
            # Create a new node (ensure uniqueness)
            b_org = T.nodes[b]["original"] if "original" in T.nodes[b] else b
            copy_b_1 = frozenset(b | frozenset(["u" + str(count)]))
            copy_b_2 = frozenset(b | frozenset(["v" + str(count)]))
            T.add_node(copy_b_1, original=b_org)
            T.add_node(copy_b_2, original=b_org)

            # Reconnect children to new nodes
            T.remove_edge(b, children[0])
            T.remove_edge(b, children[1])
            T.add_edge(copy_b_1, children[0])
            T.add_edge(copy_b_2, children[1])
            T.add_edge(b, copy_b_1)
            T.add_edge(b, copy_b_2)

            queue.append(children[0])
            queue.append(children[1])
        else:
            # Continue normal BFS for nodes with at most 2 neighbors
            for n in neighbors:
                if n not in visited:
                    queue.append(n)


def ensure_degree_2(T: networkx.Graph):
    queue = deque([frozenset()])
    visited = set()
    count = 0
    while queue:
        count += 1
        b = queue.popleft()

        if b in visited:
            continue
        visited.add(b)

        neighbors = list(T.neighbors(b))

        if len(neighbors) > 3:
            neighbor_one = neighbors[0]
            if neighbor_one in visited:
                neighbor_one = neighbors[1]
            neighbors.remove(neighbor_one)  # Remaining ones need to be moved

            # Create a new node (ensure uniqueness)
            org_b = T.nodes[b]["original"] if "original" in T.nodes[b] else b
            copy_b = frozenset(org_b | frozenset(["x" + str(count)]))  # Unique label for new node
            T.add_node(copy_b, original=org_b)

            # Transfer excess neighbors to `copy_b`
            for n in neighbors:
                if n not in visited:
                    T.remove_edge(b, n)
                    T.add_edge(copy_b, n)

            T.add_edge(b, neighbor_one)
            T.add_edge(b, copy_b)

            # Continue traversal
            queue.append(copy_b)
            queue.append(neighbor_one)

        else:
            # Continue normal BFS for nodes with at most 2 neighbors
            for n in neighbors:
                if n not in visited:
                    queue.append(n)


def __get_leaf_node(tree_decomp) -> frozenset:
    for node in tree_decomp:
        if len(tree_decomp[node]) == 1:
            return node
    return frozenset()
