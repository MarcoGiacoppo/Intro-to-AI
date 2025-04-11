import random

def generate_test_case(
    filename,
    num_nodes=6,
    coord_range=(0, 10),
    max_edges_per_node=2,
    cost_range=(1, 10),
    bidirectional_chance=0.7,
    num_destinations=2
):
    nodes = {}
    edges = {}

    # Generate random coordinates for nodes
    for node_id in range(1, num_nodes + 1):
        x = random.randint(*coord_range)
        y = random.randint(*coord_range)
        nodes[node_id] = (x, y)

    # Generate random edges
    for node in nodes:
        num_edges = random.randint(1, max_edges_per_node)
        possible_targets = list(nodes.keys())
        possible_targets.remove(node)
        targets = random.sample(possible_targets, min(num_edges, len(possible_targets)))

        for target in targets:
            cost = random.randint(*cost_range)
            edges.setdefault(node, []).append((target, cost))

            # Maybe add reverse edge
            if random.random() < bidirectional_chance:
                reverse_cost = random.randint(*cost_range)
                edges.setdefault(target, []).append((node, reverse_cost))

    # Choose random origin and destinations
    node_ids = list(nodes.keys())
    origin = random.choice(node_ids)
    destinations = random.sample([n for n in node_ids if n != origin], num_destinations)

    # Write to file
    with open(filename, "w") as f:
        f.write("Nodes:\n")
        for node_id, (x, y) in nodes.items():
            f.write(f"{node_id}: ({x},{y})\n")

        f.write("\nEdges:\n")
        for start, connections in edges.items():
            for end, cost in connections:
                f.write(f"({start},{end}): {cost}\n")

        f.write("\nOrigin:\n")
        f.write(f"{origin}\n")

        f.write("\nDestinations:\n")
        f.write("; ".join(map(str, destinations)) + "\n")

    
def generate_multiple(
    num_files=5,
    base_filename="test_case_",
    **kwargs
):
    for i in range(num_files):
        filename = f"{base_filename}{i + 1}.txt"
        generate_test_case(filename, **kwargs)
        print(f"Generated {filename}")
generate_multiple(1, "Test_")
