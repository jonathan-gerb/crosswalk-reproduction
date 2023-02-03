import dgl
import numpy as np
import torch
from crosswalk_reproduction.graphs import get_uniform_weights


import logging
logger = logging.getLogger(__name__)

def synthesize_graph(node_counts, edge_probabilities, self_connection_prob=0.0, directed=True, init_weights_strategy="uniform", weight_key="weights", remove_isolated=True, group_key="groups"):
    """Creates synthetic graph with nodes belonging to groups and edges generated according to specified probabilities.

    Args:
        node_counts (list of int): List with numbers of nodes per group. Example value [350, 150].
        edge_probabilities (torch.tensor): Tensor of shape [n_groups, n_groups] with the edge probabilities between groups.
            edge_probabilities[i,j] is the probability for an edge to exist from a node in the i-th group to one in the j-th group.
            Must be symmetric if directed=False.
            Self connections are computed seperately with self_connection_prob.
            Example value torch.tensor([[0.5, 0.01], [0.01, 0.5]])
        self_connection_prob (float, optional): Probability for self-connections. Defaults to 0.
        directed (bool, optional): Whether the graph's edges and their weights are required to be symmetric. Defaults to True.
        init_weights_strategy (bool, optional): Strategy to initialize graph.edata[weights_key]. Options:
            None: The field will not be populated.
            "uniform": Creates weights such that outgoing edge weights are initialized uniformly and sum up to 1 for each node.
            "gamma": Creates weights that follow gamma distribution with shape 1 and scale 1. 
            Defaults to "uniform".
        weight_key (str, optional): key to use for storing the edge weights, defaults to 'weights'
        remove_isolated (bool, optional): Whether nodes that are initialized without an outgoing edge should be removed.
            If used, the actual size of a group may be lower than specified in node_counts.
            Defaults to True.

    Returns:
        dgl.heterograph.DGLHeteroGraph: The resulting graph
    """
    # Type conversion
    if type(edge_probabilities) is list:
        edge_probabilities = torch.tensor(edge_probabilities)

    # Assertions
    assert edge_probabilities.shape[0] == edge_probabilities.shape[1], "edge_probabilities must be a square tensor"
    assert directed is True or (edge_probabilities == edge_probabilities.T).all(
    ), "Can not create an undirected graph with asymmetric edge_probabilities."

    # Create node data
    n_nodes = sum(node_counts)

    # bins[i] is the accumulated number of nodes with group < i
    bins = np.array([0] + [int(np.sum(node_counts[:i+1])) for i in range(len(node_counts))])

    # Create initial adjacency matrix for undirected graph with self-connections
    # (symmetry of adj for directed graphs and self-connections will be handled in later step)
    adj = torch.empty(size=(n_nodes, n_nodes))
    for source_group_idx, n_source_group in enumerate(node_counts):
        for target_group_idx, n_target_group in enumerate(node_counts):
            # Compute part of adjacency matrix with outgoing edges from source group's nodes to target group's nodes
            adj_outgoing = torch.rand(size=(n_source_group, n_target_group)
                                        ) < edge_probabilities[source_group_idx, target_group_idx]
            adj[bins[source_group_idx]: bins[source_group_idx + 1],
                bins[target_group_idx]: bins[target_group_idx + 1]] = adj_outgoing

    # If the graph is supposed to be undirected, replace upper triangle of adjacency matrics with mirrored version of lower triangle
    if directed is False:
        i, j = torch.triu_indices(n_nodes, n_nodes)
        adj[i, j] = adj.T[i, j]

    # Replace self-connections
    diag_idx = np.diag_indices(n_nodes)
    self_connections = torch.rand(size=(len(diag_idx[0]),)) < self_connection_prob
    adj[diag_idx] = self_connections.float()

    continue_removing = remove_isolated
    while continue_removing:
        continue_removing = False
        # Remove isolated nodes that do not have any outgoing edges by dropping the row and column from adj
        if remove_isolated is True:
            for node in reversed(range(adj.shape[0])):
                if (adj[node, :] == False).all():
                    # drop row
                    adj = torch.cat((adj[:node, :], adj[node+1:, :]))
                    # drop col
                    adj = torch.cat((adj[:, :node], adj[:, node+1:]), axis=1)
                    # Make sure to check another time in case removal of this node may cause another one to be isolated when dealing with directed data
                    continue_removing = False

        # Update n_nodes after removal of isolated ones
        n_nodes = adj.shape[0]

    # Convert adjacency matrix into edge indices
    edge_index = adj.nonzero().t()

    # Create graph
    graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=adj.shape[0])
    graph.ndata[group_key] = torch.tensor(np.digitize(range(n_nodes), bins))
    # initialize ids as node indices
    graph.ndata["id"] = torch.tensor(range(n_nodes))

    # Generate edge weights
    if init_weights_strategy == "uniform":
        graph.edata[weight_key] = get_uniform_weights(graph)
    elif init_weights_strategy == "gamma":
        weights = torch.tensor(np.random.gamma(1.0, 1.0, len(edge_index[0])))
        graph.edata[weight_key] = weights.type(torch.float32)
    elif init_weights_strategy == None:
        pass
    else:
        raise NotImplementedError(f"Weight initialization strategy unknown: {init_weights_strategy}.")

    _, sorted_idx = graph.ndata['id'].sort()
    for key in graph.ndata.keys():
        graph.ndata[key] = graph.ndata[key][sorted_idx]
    return graph


def get_synthetic_graph_1():
    """Creates first synthetic graph from https://arxiv.org/abs/2105.02725 (it has 2 groups).

    Returns:
        (dgl.heterograph.DGLHeteroGraph): The graph 
    """
    node_counts = [350, 150]
    edge_probabilities = torch.tensor([
        [0.025, 0.001],
        [0.001, 0.025]
    ])
    return synthesize_graph(node_counts, edge_probabilities, directed=False, init_weights_strategy="uniform")


def get_synthetic_graph_2():
    """Creates second synthetic graph from https://arxiv.org/abs/2105.02725 (it has 3 groups).

    Returns:
        (dgl.heterograph.DGLHeteroGraph): The graph 
    """
    node_counts = [300, 125, 75]
    edge_probabilities = torch.tensor([
        [0.025, 0.001, 0.0005],
        [0.001, 0.025, 0.0005],
        [0.0005, 0.0005, 0.025],
    ])

    return synthesize_graph(node_counts, edge_probabilities, directed=False, init_weights_strategy="uniform")
