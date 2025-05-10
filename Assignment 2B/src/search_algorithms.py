import heapq
from collections import deque

def dfs(start, goal, get_neighbors, cost_fn, heuristic_fn):
    stack = [(start, [start], 0)]
    visited = set()
    segment_costs = {}

    while stack:
        current, path, cost = stack.pop()
        if current == goal:
            return path, cost, segment_costs

        if current in visited:
            continue
        visited.add(current)

        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                edge_cost = cost_fn(current, neighbor)
                segment_costs[(current, neighbor)] = edge_cost
                new_cost = cost + edge_cost
                stack.append((neighbor, path + [neighbor], new_cost))

    return None, None, {}

def bfs(start, goal, get_neighbors, cost_fn, heuristic_fn):
    queue = deque([(start, [start], 0)])
    visited = set([start])
    segment_costs = {}

    while queue:
        current, path, cost = queue.popleft()
        if current == goal:
            return path, cost, segment_costs

        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                edge_cost = cost_fn(current, neighbor)
                segment_costs[(current, neighbor)] = edge_cost
                new_cost = cost + edge_cost
                queue.append((neighbor, path + [neighbor], new_cost))

    return None, None, {}

def ucs(start, goal, get_neighbors, cost_fn, heuristic_fn):
    heap = [(0, start, [start])]
    visited = {}
    segment_costs = {}

    while heap:
        cost, current, path = heapq.heappop(heap)
        if current == goal:
            return path, cost, segment_costs
        if current in visited and visited[current] <= cost:
            continue
        visited[current] = cost

        for neighbor in get_neighbors(current):
            edge_cost = cost_fn(current, neighbor)
            segment_costs[(current, neighbor)] = edge_cost
            total_cost = cost + edge_cost
            heapq.heappush(heap, (total_cost, neighbor, path + [neighbor]))

    return None, None, {}

def gbfs(start, goal, get_neighbors, cost_fn, heuristic_fn):
    heap = [(heuristic_fn(start), start, [start])]
    visited = set()
    segment_costs = {}

    while heap:
        _, current, path = heapq.heappop(heap)
        if current == goal:
            total_cost = sum(segment_costs.get((path[i], path[i+1]), 0) for i in range(len(path)-1))
            return path, total_cost, segment_costs
        if current in visited:
            continue
        visited.add(current)

        for neighbor in get_neighbors(current):
            h = heuristic_fn(neighbor)
            segment_costs[(current, neighbor)] = cost_fn(current, neighbor)
            heapq.heappush(heap, (h, neighbor, path + [neighbor]))

    return None, None, {}

def astar(start, goal, get_neighbors, cost_fn, heuristic_fn):
    heap = [(0, 0, start, [start])]
    visited = {}
    segment_costs = {}

    while heap:
        priority, cost, current, path = heapq.heappop(heap)
        if current == goal:
            return path, cost, segment_costs
        if current in visited and visited[current] <= cost:
            continue
        visited[current] = cost

        for neighbor in get_neighbors(current):
            edge_cost = cost_fn(current, neighbor)
            segment_costs[(current, neighbor)] = edge_cost
            g = cost + edge_cost
            h = heuristic_fn(neighbor)
            f = g + h
            heapq.heappush(heap, (f, g, neighbor, path + [neighbor]))

    return None, None, {}
