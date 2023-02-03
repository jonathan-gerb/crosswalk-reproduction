import dgl
import torch
from crosswalk_reproduction.graphs.graph import get_uniform_weights

import logging
logger = logging.getLogger(__name__)

def estimate_node_colorfulness(g, node_idx, walk_length, walks_per_node, group_key, prob=None):
    """Estimate proximity of a node to those of another group according to equation (3) of https://arxiv.org/abs/2105.02725.

    Args:
        g (dgl.heterograph.DGLHeteroGraph): The graph.
        node_idx (int): Index of the node for which to compute colorfulness.
        walk_length (int): The number of hops per walk.
        walks_per_node (_type_): The number of walks to sample when estimating colorfulness.
        group_key (str): Key of group labels to be used stored in g.ndata. 
        prob (str, optional): Key of weights to be used for obtaining probabilities when traversing graphs stored in g.edata. 
            Each weight has to be non-negative. If None, all weights are initialized uniformly as in the original paper. 
            Defaults to None.

    Returns:
        float: The estimated colorfulness of this node.
    """
    # Obtain walks starting from source node
    start_nodes = (torch.ones(walks_per_node) * node_idx).type(torch.int64)
    walks, _ = dgl.sampling.random_walk(g, start_nodes, length=walk_length, prob=prob)

    # Obtain groups of nodes visited
    visited_nodes = walks.flatten()
    # Drop all entries from hops where the source node had no outgoing edge
    visited_nodes = visited_nodes[visited_nodes != -1]
    visited_groups = g.ndata[group_key][visited_nodes]

    # Compute colorfulness
    colorfulness = torch.sum(visited_groups != g.ndata[group_key][node_idx]) / len(visited_nodes)

    return colorfulness.item()


def get_crosswalk_weights(g, alpha, p, walk_length, walks_per_node, group_key, prior_weights_key, use_original_crosswalk_implementation=False):
    """Computes new weights for each edge according to crosswalk strategy according to https://arxiv.org/abs/2105.02725.

    Args:
        g (dgl.heterograph.DGLHeteroGraph): The graph without parallel edges.
        alpha (float): Parameter in (0,1) to control the degree in which inter-group and intra-group connections are strengthened and weakened, respectively. High values for alpha weakens intra-group connections. 
        p (float): Parameter to control the degree of biasness of weights towards visiting nodes at group boundaries.
        walk_length (int): Length of random walks for estimating colorfulness (the degree of proximity to other groups called m in the original paper)
        walks_per_node (int): Number of random walks for estimating colorfulness (the degree of proximity to other groups called m in the original paper)
        group_key (str): Key of group labels in g.ndata which will be used by crosswalk.
        prior_weights_key (str, optional): Key of prior weights in g.edata which will be modified by crosswalk.
            Each weight has to be non-negative and each node needs at least one edge with a positive weight.
            If None, each node's outgoing weights are initialized uniformly with their sum being normalized to 1.
            This guarantees that each nodes resulting weights also add up to 1.
            Defaults to None.
        use_original_crosswalk_implementation (bool, optional): Whether to use the implementation details according to the original paper's github repository.
            If True, 0.001 will be added to each nodes' colorfulness.
            If False, edge cases with colorfulnesses of 0 are possible and will be handled according to the equations proposed during the reproducibility challenge.
            Defaults to False.

    Returns:
        torch.tensor: Tensor with one weight per edge, ordered according to id's of g.edges.
    """
    assert not g.is_multigraph, "Crosswalk reweighting does not support parallel edges."
    assert 0 < alpha < 1, f"alpha needs to be in (0,1). Received {alpha=}"
    assert 0 < p, f"p needs to greater than 0. Received {p=}"
    assert prior_weights_key is None or (g.edata[prior_weights_key] > 0).all(), "If provided, prior weights have to be larger than 0."

    # Initialize weights if not provided
    if prior_weights_key is not None:
        prior_weights = g.edata[prior_weights_key]
    else:
        prior_weights = get_uniform_weights(g)

    # Pre-Compute colorfulness and normalization factors from formula (4) for each node
    colorfulnesses = torch.tensor([estimate_node_colorfulness(
        g, node, walk_length, walks_per_node, group_key, prob=None) for node in g.nodes()])

    # In the original implementation, 0.001 was added to all colorfulness estimates
    # This avoided edge cases when colorfulnesses are 0
    # These edge cases can be handled by this implementation, making the addition of 0.001 optional
    if use_original_crosswalk_implementation:
        colorfulnesses = colorfulnesses + 0.001

    # Compute new weights
    new_weights = torch.empty_like(prior_weights)
    for source in g.nodes():
        source_group = g.ndata[group_key][source]
        all_neighbors = g.successors(source)
        neighbor_groups = g.ndata[group_key][all_neighbors]
        same_group_neighbors = all_neighbors[neighbor_groups == source_group]
        n_different_groups_in_neighborhood = len(neighbor_groups[neighbor_groups != source_group].unique())

        # Handle edges towards nodes of same group
        # Compute normalization factor for intra-group connections


        for group in neighbor_groups.unique():
            group_neighbors = all_neighbors[neighbor_groups == group]
            z = sum([(colorfulnesses[nb.item()] ** p) * prior_weights[g.edge_ids(source, nb)]
                    for nb in group_neighbors])

            # Compute weights for all neighboring nodes of same group
            for nb in group_neighbors:
                prior_weight = prior_weights[g.edge_ids(source, nb)]

                # Compute n with either equation 2a or 2b
                if z > 0:
                    n = prior_weight * (colorfulnesses[nb.item()] ** p) / z
                else:
                    total_prior_weights_towards_group = sum([prior_weights[g.edge_ids(source, nb)] for nb in group_neighbors])
                    n = prior_weight / total_prior_weights_towards_group

                # Compute weights towards neighbors of the same color with either equation 3a or 4a
                if group == source_group:
                    # Equation 3a, if neighbors exist neighbors of different color
                    if len(group_neighbors) < len(all_neighbors):
                        new_weights[g.edge_ids(source, nb)] = (1 - alpha) * n
                    # Equation 4a, otherwise
                    else:
                        new_weights[g.edge_ids(source, nb)] = n

                # Compute weights towards neighbors of different colors with either equation 3b or 4b
                else:
                    # Equation 3b, if v has at least one neighbor of the same color
                    if len(same_group_neighbors) > 0:
                        new_weights[g.edge_ids(source, nb)] = alpha * n / n_different_groups_in_neighborhood
                    # Equation 4b, otherwise
                    else:
                        new_weights[g.edge_ids(source, nb)] = n / n_different_groups_in_neighborhood
    return new_weights
