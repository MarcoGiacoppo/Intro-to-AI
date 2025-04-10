import pyglet
from pyglet import shapes

class Visualizer:
    def __init__(self, nodes, edges, path):
        self.nodes = nodes
        self.edges = edges
        self.path = path

        self.window = pyglet.window.Window(800, 600, "Pathfinding Visualizer")
        self.batch = pyglet.graphics.Batch()
        self.labels = []

        self.node_radius = 12
        self.node_shapes = {}
        self.edge_shapes = []
        
        # Add padding as a percentage of window size
        self.padding = 0.1  # 10% padding on all sides
        
        self.prepare_graph()

    def scale_coordinates(self):
        # Find min and max coordinates
        min_x = min(x for x, y in self.nodes.values())
        max_x = max(x for x, y in self.nodes.values())
        min_y = min(y for x, y in self.nodes.values())
        max_y = max(y for x, y in self.nodes.values())
        
        # Calculate dimensions of the original graph
        graph_width = max_x - min_x
        graph_height = max_y - min_y
        
        # Calculate available window space with padding
        window_width = self.window.width * (1 - 2 * self.padding)
        window_height = self.window.height * (1 - 2 * self.padding)
        
        # Calculate scaling factor (preserving aspect ratio)
        scale = min(window_width / graph_width, window_height / graph_height)
        
        # Calculate offsets to center the graph
        offset_x = (self.window.width - (graph_width * scale)) / 2
        offset_y = (self.window.height - (graph_height * scale)) / 2
        
        # Create scaled node coordinates
        scaled_nodes = {}
        for node_id, (x, y) in self.nodes.items():
            new_x = offset_x + (x - min_x) * scale
            new_y = offset_y + (y - min_y) * scale
            scaled_nodes[node_id] = (new_x, new_y)
        
        return scaled_nodes

    def prepare_graph(self):
        # Get scaled coordinates
        scaled_nodes = self.scale_coordinates()
        
        # Draw edges
        for a, b in self.edges:
            if a in scaled_nodes and b in scaled_nodes:
                x1, y1 = scaled_nodes[a]
                x2, y2 = scaled_nodes[b]
                self.edge_shapes.append(
                    shapes.Line(x1, y1, x2, y2, thickness=2, color=(150, 150, 150), batch=self.batch)
                )

        # Draw nodes + labels
        for node_id, (x, y) in scaled_nodes.items():
            circle = shapes.Circle(x, y, self.node_radius, color=(100, 180, 250), batch=self.batch)
            self.node_shapes[node_id] = circle

            label = pyglet.text.Label(
                str(node_id),
                font_size=12,
                x=x,
                y=y + self.node_radius + 5,
                anchor_x='center',
                batch=self.batch
            )
            self.labels.append(label)

        # Highlight path edges in red
        for i in range(len(self.path) - 1):
            a = self.path[i]
            b = self.path[i + 1]
            if a in scaled_nodes and b in scaled_nodes:
                x1, y1 = scaled_nodes[a]
                x2, y2 = scaled_nodes[b]
                self.edge_shapes.append(
                    shapes.Line(x1, y1, x2, y2, thickness=4, color=(255, 50, 50), batch=self.batch)
                )

    def run(self):
        @self.window.event
        def on_draw():
            self.window.clear()
            self.batch.draw()

        pyglet.app.run()