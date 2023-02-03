import torch

def get_uniform_weights(g):
    """Creates weights such that outgoing edge weights are initialized uniformly and sum up to 1 for each node.

    Args:
        g (dgl.heterograph.DGLHeteroGraph): The graph

    Returns:
        torch.tensor: Tensor with one weight per edge in g.edges()
    """
    normalized_weights = []
    for e in g.edges(form="eid"):
        source_node = g.edges()[0][e]
        weight = 1 / g.out_degrees(source_node)
        normalized_weights += [weight]
    weights = torch.tensor(normalized_weights) 
    return weights