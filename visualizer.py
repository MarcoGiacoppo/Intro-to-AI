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

        self.prepare_graph()

    def prepare_graph(self):
        # Draw edges
        for a, b in self.edges:
            if a in self.nodes and b in self.nodes:
                x1, y1 = self.nodes[a]
                x2, y2 = self.nodes[b]
                self.edge_shapes.append(
                    shapes.Line(x1, y1, x2, y2, thickness=2, color=(150, 150, 150), batch=self.batch)
                )

        # Draw nodes + labels
        for node_id, (x, y) in self.nodes.items():
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
            if a in self.nodes and b in self.nodes:
                x1, y1 = self.nodes[a]
                x2, y2 = self.nodes[b]
                self.edge_shapes.append(
                    shapes.Line(x1, y1, x2, y2, thickness=4, color=(255, 50, 50), batch=self.batch)
                )

    def run(self):
        @self.window.event
        def on_draw():
            self.window.clear()
            self.batch.draw()

        pyglet.app.run()
