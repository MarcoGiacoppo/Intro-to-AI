from visualizer import Visualizer
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
                        scale = 80
                        self.nodes[int(node_id)] = (x * scale, y * scale)

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
    # The stack keeps a tuple of the current node and the path taken to reach it

    visited = set()
    node_count = 0  # Count of created nodes (i.e., added to stack)

    while stack:
        node, path = stack.pop()
        # Repeated state checks
        if node in visited:
            continue
        visited.add(node)

        if node in goals:
            return node, node_count, path, visited  # Goal found

        # Add neighbors to stack (sorted for consistent expansion)
        for neighbor, _ in sorted(graph.edges.get(node, [])): # _ means to ignore the cost. (node, []) means to get the value from the node key, and return nothing if it doesnt exist. 
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor])) #append the edge node and path taken to reach it. **path + [neighbor] creates a new list instead of modifying the older list. 
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
        node, path = queue.popleft() # first in first out by using popleft() method.

        if node in visited:
            continue
        visited.add(node)

        if node in goals:
            return node, node_count, path, visited  # Goal found

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
        ) #calculates the distance by pythagorean theorem. 

def gbfs(graph, start, goals):
    frontier = []
    heapq.heappush(frontier, (heuristic(graph, start, goals), start, [start], 0)) #Automatically sorts smalles first into the frontier list. 
    visited = set()
    node_count = 0

    while frontier:
        _, current, path, cost = heapq.heappop(frontier) # pop the node with the lowest heuristic value.

        if current in visited:
            continue
        visited.add(current)

        if current in goals:
            return current, node_count, path, cost, visited

        for neighbor, Ncost in sorted(graph.edges.get(current, [])):
            if neighbor not in visited:
                h = heuristic(graph, neighbor, goals)
                heapq.heappush(frontier, (h, neighbor, path + [neighbor], cost + Ncost))
                node_count += 1

    return None, node_count, None

# ---------
# A* Search
# ---------
def astar(graph, start, goals):
    frontier = []
    heapq.heappush(frontier, (heuristic(graph, start, goals), 0, start, [start], 0))  # (f, g, node, path)
    visited = set()
    node_count = 0

    # f = g + h, where g is the cost to reach the node and h is the heuristic

    while frontier:
        _, g, current, path, cost = heapq.heappop(frontier)

        if current in visited:
            continue
        visited.add(current)

        if current in goals:
            return current, node_count, path, cost, visited

        # This code block here shows that we're mainting g (path cost) and f = g + h
        for neighbor, Ncost in sorted(graph.edges.get(current, [])):
            if neighbor not in visited:
                g_new = g + Ncost
                h = heuristic(graph, neighbor, goals)
                f_new = g_new + h
                heapq.heappush(frontier, (f_new, g_new, neighbor, path + [neighbor], cost + Ncost))
                node_count += 1

    return None, node_count, None


# ----------------------------------------------------------------------------
# CUS2: Informed, A* where g(n) is number of steps taken, and h(n) is the euclidean distance to the goal.
# ----------------------------------------------------------------------------
def cus2(graph, start, goals): #may not be allowed
    frontier = []
    heapq.heappush(frontier, (heuristic(graph, start, goals), 0, start, [start], 0))  # (f, g, node, path)
    visited = set()
    node_count = 0

    # f = g + h, where g is the cost to reach the node and h is the heuristic

    while frontier:
        _, g, current, path, cost = heapq.heappop(frontier)

        if current in visited:
            continue
        visited.add(current)

        if current in goals:
            return current, node_count, path, cost, visited

        # This code block here shows that we're mainting g (path cost) and f = g + h
        for neighbor, Ncost in sorted(graph.edges.get(current, [])):
            if neighbor not in visited:
                g_new = g + 1
                h = heuristic(graph, neighbor, goals)
                f_new = g_new + h
                heapq.heappush(frontier, (f_new, g_new, neighbor, path + [neighbor], cost + Ncost))
                node_count += 1

    return None, node_count, None



# -----------------------------------------------------------------------------------------
# CUS1: Uniform Cost Search - prefer lower cost paths first
# -----------------------------------------------------------------------------------------
def cus1(graph, start, goals):
    frontier = []
    heapq.heappush(frontier, (0, start, [start], 0))  # (g, node, path)
    visited = set()
    node_count = 0

    while frontier:
        g, current, path, cost = heapq.heappop(frontier)

        if current in visited:
            continue
        visited.add(current)

        if current in goals:
            return current, node_count, path, cost, visited

        for neighbor, Ncost in sorted(graph.edges.get(current, [])):
            if neighbor not in visited:
                g_new = g + Ncost
                heapq.heappush(frontier, (g_new, neighbor, path + [neighbor], cost + Ncost))
                node_count += 1

    return None, node_count, None


# --------------------------------------
# Main Execution (Command-Line Interface)
# --------------------------------------
import sys
import time
if __name__ == "__main__":
    # Check correct number of arguments
    if len(sys.argv) < 3:
        print("Usage: python search.py <filename> <method> [--visualize] [debug]")
        sys.exit(1)

    filename = sys.argv[1]
    method = sys.argv[2].upper()  # e.g., BFS, DFS
    visualize = "--visualize" in sys.argv
    debug = "debug" in sys.argv

    # Load graph from file
    graph = Graph()
    graph.load_from_file(filename)
    
    # Run the selected search method
    cost= 0
    start=time.perf_counter()  # Start timing
    if method == "DFS":
        goal, count, path, visited = dfs(graph, graph.origin, graph.destinations)
    elif method == "BFS":
        goal, count, path, visited = bfs(graph, graph.origin, graph.destinations)
    elif method == "GBFS":
        goal, count, path, cost, visited = gbfs(graph, graph.origin, graph.destinations)
    elif method == "AS":
        goal, count, path, cost, visited = astar(graph, graph.origin, graph.destinations)
    elif method == "CUS1":
        goal, count, path, cost, visited = cus1(graph, graph.origin, graph.destinations)
    elif method == "CUS2":
        goal, count, path, cost, visited = cus2(graph, graph.origin, graph.destinations)
    else:
        print("Unsupported method. Use DFS, BFS, GBFS, or AS.")
        sys.exit(1)
    end = time.perf_counter()  # End timing
    elapsed_time = end - start  # Calculate elapsed time
    # Print the output in the required format
    if path:
        print(f"{filename} {method}")
        print(f"{goal} {count}")
        print(" -> ".join(map(str, path)))

        # Prepare edges in (start, end) format for visualizer
        edge_list = []
        for start, connections in graph.edges.items():
            for end, _ in connections:
                edge_list.append((start, end))

        # Run visualizer
        if visualize:
            Visualizer(graph.nodes, edge_list, path).run()
        
        if debug:
            print("Debug Information:")
            print(f"Visited nodes: {visited}")
            print(f"Path cost: {cost}")
            print(f"Time taken: {elapsed_time:.8f} seconds")
    else:
        print(f"{filename} {method}")
        print("No path found.")


# ------------------------
# HOW TO RUN THIS PROGRAM
# ------------------------

# In your terminal or command prompt:
# python search.py PathFinder-test.txt BFS
# or
# python search.py PathFinder-test.txt DFS
# -----------
# Extensions 
# -----------
# pyglet needs to be installed for visualisation: (pip install --upgrade --user pyglet)
# Try using pyglet to visualize the nodes visited
# python search.py Test_1.txt BFS --visualize