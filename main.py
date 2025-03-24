# ------------------------
# Graph Representation
# ------------------------
class Graph:
    def __init__(self):
        # Dictionary of nodes: {node_id: (x, y)}
        self.nodes = {}
        # Dictionary of edges: {start_node: [(end_node, cost), ...]}
        self.edges = {}
        self.origin = None
        self.destinations = []

    def load_from_file(self, filename):
        """Loads nodes, edges, origin, and destinations from the file."""
        with open(filename, 'r') as file:
            section = None
            for line in file:
                line = line.strip()
                if not line:
                    continue

                # Identify current section of the file
                if line.startswith("Nodes:"):
                    section = "nodes"
                elif line.startswith("Edges:"):
                    section = "edges"
                elif line.startswith("Origin:"):
                    section = "origin"
                elif line.startswith("Destinations:"):
                    section = "destinations"
                else:
                    # Parse each section accordingly
                    if section == "nodes":
                        node_id, coords = line.split(": ")
                        x, y = map(int, coords.strip("()").split(","))
                        self.nodes[int(node_id)] = (x, y)

                    elif section == "edges":
                        edge, cost = line.split(": ")
                        start, end = map(int, edge.strip("()").split(","))
                        self.edges.setdefault(start, []).append((end, int(cost)))

                    elif section == "origin":
                        self.origin = int(line)

                    elif section == "destinations":
                        self.destinations = list(map(int, line.split(";")))

    def display_graph(self):
        """Prints the internal structure of the graph (for debugging)."""
        print(f"Nodes: {self.nodes}")
        print(f"Edges: {self.edges}")
        print(f"Origin: {self.origin}")
        print(f"Destinations: {self.destinations}")


# ------------------------
# Depth-First Search (DFS)
# ------------------------
def dfs(graph, start, goals):
    stack = [(start, [start])]  # Stack stores (current_node, path)
    visited = set()
    node_count = 0  # Count of created nodes (i.e., added to stack)

    while stack:
        node, path = stack.pop()

        if node in visited:
            continue
        visited.add(node)

        if node in goals:
            return node, node_count, path  # Goal found

        # Add neighbors to stack (sorted for consistent expansion)
        for neighbor, _ in sorted(graph.edges.get(node, [])):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))
                node_count += 1  # Increment when node is added to stack

    return None, node_count, None  # No goal found


# ------------------------
# Breadth-First Search (BFS)
# ------------------------
from collections import deque

def bfs(graph, start, goals):
    queue = deque([(start, [start])])  # Queue stores (current_node, path)
    visited = set()
    node_count = 0

    while queue:
        node, path = queue.popleft()

        if node in visited:
            continue
        visited.add(node)

        if node in goals:
            return node, node_count, path  # Goal found

        # Add neighbors to queue (sorted for consistent expansion)
        for neighbor, _ in sorted(graph.edges.get(node, [])):
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))
                node_count += 1  # Increment when node is added to queue

    return None, node_count, None  # No goal found


# ------------------------
# Main Execution (Command-Line Interface)
# ------------------------
import sys

if __name__ == "__main__":
    # Check correct number of arguments
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        sys.exit(1)

    filename = sys.argv[1]
    method = sys.argv[2].upper()  # e.g., BFS, DFS

    # Load graph from file
    graph = Graph()
    graph.load_from_file(filename)

    # Run the selected search method
    if method == "DFS":
        goal, count, path = dfs(graph, graph.origin, graph.destinations)
    elif method == "BFS":
        goal, count, path = bfs(graph, graph.origin, graph.destinations)
    else:
        print("Unsupported method. Use DFS or BFS.")
        sys.exit(1)

    # Print the output in the required format
    if path:
        print(f"{filename} {method}")
        print(f"{goal} {count}")
        print(" -> ".join(map(str, path)))
    else:
        print(f"{filename} {method}")
        print("No path found.")


# ------------------------
# HOW TO RUN THIS PROGRAM
# ------------------------
# In your terminal or command prompt:
# python3 search.py PathFinder-test.txt BFS
# or
# python3 search.py PathFinder-test.txt DFS
