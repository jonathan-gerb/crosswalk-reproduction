import torch

import logging
logger = logging.getLogger(__name__)

def get_fairwalk_weights(g, group_key):
    """Computes new weights for each edgs according to fairwalk strategy according to https://www.ijcai.org/proceedings/2019/0456.pdf. 

    This means that each node's outgoing edge weights are initialized such that
    (i) they sum up to one, and
    (ii) the sum of edge weights toward its members should be the same for all groups, and
    (iii) each edge towards a node of the same group should receive the same weight.

    Args:
        g (dgl.heterograph.DGLHeteroGraph): The graph without parallel edges.
        group_key (str): Key of group labels in g.ndata which will be used by fairwalk.

    Returns:
        torch.tensor: Tensor with one weight per edge
    """

    assert not g.is_multigraph, "Fairwalk reweighting does not support parallel edges."

    # Iterate through all nodes and compute their outgoing edges' weights
    new_weights = torch.empty(size=(g.num_edges(),))
    for source in g.nodes():
        all_neighbors = g.successors(source)
        neighbor_groups = g.ndata[group_key][all_neighbors]
        unique_neighbor_groups = neighbor_groups.unique()

        # No neighbor no cry
        if len(unique_neighbor_groups) == 0:
            continue

        # the sum of edge weights toward its members should be the same for all groups
        total_weight_per_group = 1.0 / len(unique_neighbor_groups)

        for group in unique_neighbor_groups:
            group_neighbors = all_neighbors[neighbor_groups == group]
            # Each edge towards a node of the same group should receive the same weight
            weight_per_node = total_weight_per_group / len(group_neighbors)

            for nb in group_neighbors:
                new_weights[g.edge_ids(source, nb)] = weight_per_node

    return new_weights
