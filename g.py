import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Define nodes
nodes = {
    "input": "Input vector\n$v$",
    "W1": "Weights\n$W^{(1)}$",
    "b1": "Bias\n$b^{(1)}$",
    "linear1": "Linear 1:\n$W^{(1)}v + b^{(1)}$",
    "sigmoid1": "Sigmoid 1:\n$z = \\sigma(\\cdot)$",
    "W2": "Weights\n$W^{(2)}$",
    "b2": "Bias\n$b^{(2)}$",
    "linear2": "Linear 2:\n$W^{(2)}z + b^{(2)}$",
    "sigmoid2": "Sigmoid 2:\n$\\hat{v} = \\sigma(\\cdot)$",
    "target": "Target vector\n$v$ (known entries)",
    "loss": "Loss:\n$\\sum (\\hat{v} - v)^2$"
}

# Add nodes to graph
for k, v in nodes.items():
    G.add_node(k, label=v)

# Define edges
edges = [
    ("input", "linear1"), ("W1", "linear1"), ("b1", "linear1"),
    ("linear1", "sigmoid1"),
    ("sigmoid1", "linear2"), ("W2", "linear2"), ("b2", "linear2"),
    ("linear2", "sigmoid2"),
    ("sigmoid2", "loss"),
    ("target", "loss")
]

G.add_edges_from(edges)

# Position using graphviz layout
pos = nx.nx_agraph.graphviz_layout(G, prog='dot')

# Draw the graph
plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=False, arrows=True, node_size=3000, node_color="#D6EAF8")
nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'), font_size=10)
plt.title("Computation Graph for Autoencoder Training", fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.show()
