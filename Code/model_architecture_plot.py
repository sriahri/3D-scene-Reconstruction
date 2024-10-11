# import pygraphviz as pgv

# # Create a new graph
# G = pgv.AGraph(directed=True, strict=True)

# # Add nodes with specific attributes
# nodes = [
#     ("Input", {'shape': 'box'}),
#     ("Encoder", {'shape': 'box'}),
#     ("Residual Blocks", {'shape': 'box'}),
#     ("Decoder", {'shape': 'box'}),
#     ("UpConv6", {'shape': 'box'}),
#     ("Iconv6", {'shape': 'box'}),
#     ("UpConv5", {'shape': 'box'}),
#     ("Iconv5", {'shape': 'box'}),
#     ("UpConv4", {'shape': 'box'}),
#     ("Iconv4", {'shape': 'box'}),
#     ("Disp4 Layer", {'shape': 'box'}),
#     ("UpConv3", {'shape': 'box'}),
#     ("Iconv3", {'shape': 'box'}),
#     ("Disp3 Layer", {'shape': 'box'}),
#     ("UpConv2", {'shape': 'box'}),
#     ("Iconv2", {'shape': 'box'}),
#     ("Disp2 Layer", {'shape': 'box'}),
#     ("UpConv1", {'shape': 'box'}),
#     ("Iconv1", {'shape': 'box'}),
#     ("Disp1 Layer", {'shape': 'box'})
# ]

# # Add nodes to the graph
# for node in nodes:
#     G.add_node(node[0], **node[1])

# # Define edges
# edges = [
#     ("Input", "Encoder"),
#     ("Encoder", "Residual Blocks"),
#     ("Residual Blocks", "Decoder"),
#     ("Decoder", "UpConv6"),
#     ("UpConv6", "Iconv6"),
#     ("Iconv6", "UpConv5"),
#     ("UpConv5", "Iconv5"),
#     ("Iconv5", "UpConv4"),
#     ("UpConv4", "Iconv4"),
#     ("Iconv4", "Disp4 Layer"),
#     ("Disp4 Layer", "UpConv3"),
#     ("UpConv3", "Iconv3"),
#     ("Iconv3", "Disp3 Layer"),
#     ("Disp3 Layer", "UpConv2"),
#     ("UpConv2", "Iconv2"),
#     ("Iconv2", "Disp2 Layer"),
#     ("Disp2 Layer", "UpConv1"),
#     ("UpConv1", "Iconv1"),
#     ("Iconv1", "Disp1 Layer")
# ]

# # Add edges to the graph
# for edge in edges:
#     G.add_edge(edge[0], edge[1])

# # Draw the graph horizontally (e.g., left to right)
# output_file_horizontal = "model_architecture_horizontal.png"
# G.draw(output_file_horizontal, prog='dot')

# print(f"Horizontal model architecture diagram saved as '{output_file_horizontal}'")

# # Draw the graph vertically (e.g., top to bottom)
# output_file_vertical = "model_architecture_vertical.png"
# G.draw(output_file_vertical, prog='twopi', args='-Grankdir=TB')

# print(f"Vertical model architecture diagram saved as '{output_file_vertical}'")
# import graphviz

# # Create a new graph
# graph = graphviz.Digraph(format='png', engine='dot')

# # Define nodes and edges
# graph.node('Input (Image)')
# graph.node('Encoder (CNN Layers)')
# graph.node('Decoder (Upsampling Layers)')
# graph.node('Depth Estimation (Output)')

# # Define connections between nodes
# graph.edge('Input (Image)', 'Encoder (CNN Layers)', label='Image Features')
# graph.edge('Encoder (CNN Layers)', 'Decoder (Upsampling Layers)', label='Feature Maps')
# graph.edge('Decoder (Upsampling Layers)', 'Depth Estimation (Output)', label='Depth Map')

# # Render and display the graph
# graph.render('model_architecture', view=True)

from graphviz import Digraph

# Create a new directed graph
G = Digraph('Model Architecture', filename='model_architecture', format='png')



# Define nodes with specific attributes
nodes = [
    ("Input", {'shape': 'box'}),
    ("Encoder", {'shape': 'box'}),
    ("Residual Blocks", {'shape': 'box'}),
    ("Decoder", {'shape': 'box'}),
    ("UpConv6", {'shape': 'box'}),
    ("Iconv6", {'shape': 'box'}),
    ("UpConv5", {'shape': 'box'}),
    ("Iconv5", {'shape': 'box'}),
    ("UpConv4", {'shape': 'box'}),
    ("Iconv4", {'shape': 'box'}),
    ("Disp4 Layer", {'shape': 'box'}),
    ("UpConv3", {'shape': 'box'}),
    ("Iconv3", {'shape': 'box'}),
    ("Disp3 Layer", {'shape': 'box'}),
    ("UpConv2", {'shape': 'box'}),
    ("Iconv2", {'shape': 'box'}),
    ("Disp2 Layer", {'shape': 'box'}),
    ("UpConv1", {'shape': 'box'}),
    ("Iconv1", {'shape': 'box'}),
    ("Disp1 Layer", {'shape': 'box'})
]

# Define subgraphs for horizontal arrangement
with G.subgraph() as cluster_input:
    cluster_input.attr(rank='same')
    for node in nodes[:4]:  # Nodes from "Input" to "Decoder"
        cluster_input.node(node[0], **node[1])

with G.subgraph() as cluster_decoder:
    cluster_decoder.attr(rank='same')
    for node in nodes[4:8]:  # Nodes from "UpConv6" to "Iconv5"
        cluster_decoder.node(node[0], **node[1])

with G.subgraph() as cluster_disp:
    cluster_disp.attr(rank='same')
    for node in nodes[8:11]:  # Nodes from "UpConv4" to "Disp1 Layer"
        cluster_disp.node(node[0], **node[1])

with G.subgraph() as cluster_disp:
    cluster_disp.attr(rank='same')
    for node in nodes[11:14]:  # Nodes from "UpConv4" to "Disp1 Layer"
        cluster_disp.node(node[0], **node[1])
with G.subgraph() as cluster_disp:
    cluster_disp.attr(rank='same')
    for node in nodes[14:17]:  # Nodes from "UpConv4" to "Disp1 Layer"
        cluster_disp.node(node[0], **node[1])

with G.subgraph() as cluster_disp:
    cluster_disp.attr(rank='same')
    for node in nodes[17:]:  # Nodes from "UpConv4" to "Disp1 Layer"
        cluster_disp.node(node[0], **node[1])




# Define edges
edges = [
    ("Input", "Encoder"),
    ("Encoder", "Residual Blocks"),
    ("Residual Blocks", "Decoder"),
    ("Decoder", "UpConv6"),
    ("UpConv6", "Iconv6"),
    ("Iconv6", "UpConv5"),
    ("UpConv5", "Iconv5"),
    ("Iconv5", "UpConv4"),
    ("UpConv4", "Iconv4"),
    ("Iconv4", "Disp4 Layer"),
    ("Disp4 Layer", "UpConv3"),
    ("UpConv3", "Iconv3"),
    ("Iconv3", "Disp3 Layer"),
    ("Disp3 Layer", "UpConv2"),
    ("UpConv2", "Iconv2"),
    ("Iconv2", "Disp2 Layer"),
    ("Disp2 Layer", "UpConv1"),
    ("UpConv1", "Iconv1"),
    ("Iconv1", "Disp1 Layer")
]

# Add edges to the graph
for edge in edges:
    G.edge(edge[0], edge[1])

# Render and save the graph as an image
G.render(view=True)