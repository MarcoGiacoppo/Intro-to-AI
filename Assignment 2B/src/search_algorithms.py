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
                new_cost = cost + edge_cost + 0.5
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
                new_cost = cost + edge_cost + 0.5
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

def bidirectional(start, goal, get_neighbors, cost_fn=None, heuristic_fn=None):
    from collections import deque

    start, goal = str(start), str(goal)
    if start == goal:
        return [start], 0.0, {}

    frontier_start = deque([start])
    frontier_goal = deque([goal])

    visited_start = {start: None}
    visited_goal = {goal: None}

    meet_node = None
    segment_costs = {}

    while frontier_start and frontier_goal:
        # Expand forward
        current_start = frontier_start.popleft()
        for neighbor in get_neighbors(current_start):
            if neighbor not in visited_start:
                visited_start[neighbor] = current_start
                frontier_start.append(neighbor)
                if neighbor in visited_goal:
                    meet_node = neighbor
                    break
        if meet_node:
            break

        # Expand backward
        current_goal = frontier_goal.popleft()
        for neighbor in get_neighbors(current_goal):
            if neighbor not in visited_goal:
                visited_goal[neighbor] = current_goal
                frontier_goal.append(neighbor)
                if neighbor in visited_start:
                    meet_node = neighbor
                    break
        if meet_node:
            break

    if not meet_node:
        return None, float("inf"), {}

    # === Reconstruct the path in-place ===
    path = []

    # Backtrack from meet_node to start
    node = meet_node
    while node:
        path.insert(0, node)
        node = visited_start[node]

    # Forward from meet_node to goal
    node = visited_goal[meet_node]
    while node:
        path.append(node)
        node = visited_goal[node]

    # === Optional: Track segment costs ===
    total_cost = 0.0
    for i in range(1, len(path)):
        a, b = path[i - 1], path[i]
        if cost_fn:
            c = cost_fn(a, b)
        else:
            c = 1
        total_cost += c
        segment_costs[(a, b)] = c

    return path, round(total_cost, 2), segment_costs
