import networkx as nx
from dgl.heterograph import DGLHeteroGraph
import matplotlib.pyplot as plt


def visualize_graph(g, node_color_key=None, edge_label_key=None):
    """Visualizes graph

    Args:
        g (nx or torch geometric graph): The graph to visualize.
        node_color_key (str, optional): If provided and g is DGLHeteroGraph, then node colors will be plotted according to g.ndata[node_color_key]. Defaults to None.
        edge_label_key (str, optional): If provided and g is DGLHeteroGraph, then edge labels will be plotted according to g.edata[edge_label_key]. Defaults to None.
    """

    # Convert data to nx graph if necessary
    if type(g) in [nx.classes.graph.Graph, nx.classes.digraph.DiGraph]:
        g = g
    elif type(g) in [DGLHeteroGraph]:
        node_color = None if node_color_key is None else g.ndata[node_color_key]
        edge_labels = None
        if edge_label_key is not None:
            edge_labels = {}
            for idx, edge in enumerate(g.edges(form="eid")):
                source_node_id = g.edges()[0][idx].item()
                target_node_id = g.edges()[1][idx].item()
                label = g.edata[edge_label_key][idx].item()
                if type(label) is float:
                    label = "{:.2f}".format(label)
                edge_labels[(source_node_id, target_node_id)] = label
        g = g.to_networkx()

    plt.figure()
    pos = nx.spring_layout(g)
    nx.draw_networkx(g, pos, node_color=node_color)
    if edge_labels is not None:
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, label_pos=0.8, font_size=5)
