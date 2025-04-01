import math
import heapq

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
        # Repeated state checks
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

# ------------------------------
# Greedy Best-Frist Search (GBFS)
# ------------------------------
def heuristic(graph, node, goals):
        """Calculate the Euclidean distance to the nearest goal."""
        x1, y1 = graph.nodes[node]
        return min(
            math.sqrt((x1 - graph.nodes[g][0])**2 + (y1 - graph.nodes[g][1])**2)
            for g in goals
        )

def gbfs(graph, start, goals):
    frontier = []
    heapq.heappush(frontier, (heuristic(graph, start, goals), start, [start]))
    visited = set()
    node_count = 0

    while frontier:
        _, current, path = heapq.heappop(frontier)

        if current in visited:
            continue
        visited.add(current)

        if current in goals:
            return current, node_count, path

        for neighbor, _ in sorted(graph.edges.get(current, [])):
            if neighbor not in visited:
                h = heuristic(graph, neighbor, goals)
                heapq.heappush(frontier, (h, neighbor, path + [neighbor]))
                node_count += 1

    return None, node_count, None

# ---------
# A* Search
# ---------
def astar(graph, start, goals):
    frontier = []
    heapq.heappush(frontier, (heuristic(graph, start, goals), 0, start, [start]))  # (f, g, node, path)
    visited = set()
    node_count = 0

    while frontier:
        f, g, current, path = heapq.heappop(frontier)

        if current in visited:
            continue
        visited.add(current)

        if current in goals:
            return current, node_count, path

        # This code block here shows that we're mainting g (path cost) and f = g + h
        for neighbor, cost in sorted(graph.edges.get(current, [])):
            if neighbor not in visited:
                g_new = g + cost
                h = heuristic(graph, neighbor, goals)
                f_new = g_new + h
                heapq.heappush(frontier, (f_new, g_new, neighbor, path + [neighbor]))
                node_count += 1

    return None, node_count, None


# ----------------------------------------------------------------------------
# CUS1: Reverse-BFS - prefer deeper paths first (a hybrid between DFS and BFS)
# ----------------------------------------------------------------------------
def cus1(graph, start, goals):
    stack = [(start, [start])]
    visited = set()
    node_count = 0

    while stack:
        node, path = stack.pop(0)  # pop from front to mimic 'oldest-deepest'

        if node in visited:
            continue
        visited.add(node)

        if node in goals:
            return node, node_count, path

        for neighbor, _ in reversed(sorted(graph.edges.get(node, []))):
            if neighbor not in visited:
                stack.insert(0, (neighbor, path + [neighbor]))  # insert at front
                node_count += 1

    return None, node_count, None


# -----------------------------------------------------------------------------------------
# CUS2: Prioritize lowest cost path with slight bias toward goal proximity (but not full A*)
# -----------------------------------------------------------------------------------------
def cus2(graph, start, goals):
    frontier = []
    heapq.heappush(frontier, (0, heuristic(graph, start, goals), start, [start]))  # (g, h, node, path)
    visited = set()
    node_count = 0

    while frontier:
        g, h, current, path = heapq.heappop(frontier)

        if current in visited:
            continue
        visited.add(current)

        if current in goals:
            return current, node_count, path

        for neighbor, cost in sorted(graph.edges.get(current, [])):
            if neighbor not in visited:
                g_new = g + cost
                h_new = heuristic(graph, neighbor, goals)
                heapq.heappush(frontier, (g_new, h_new, neighbor, path + [neighbor]))
                node_count += 1

    return None, node_count, None


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
    elif method == "GBFS":
        goal, count, path = gbfs(graph, graph.origin, graph.destinations)
    elif method == "AS":
        goal, count, path = astar(graph, graph.origin, graph.destinations)
    elif method == "CUS1":
        goal, count, path = cus1(graph, graph.origin, graph.destinations)
    elif method == "CUS2":
        goal, count, path = cus2(graph, graph.origin, graph.destinations)
    else:
        print("Unsupported method. Use DFS, BFS, GBFS, or AS.")
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
# python3 main.py PathFinder-test.txt BFS
# or
# python3 main.py PathFinder-test.txt DFS