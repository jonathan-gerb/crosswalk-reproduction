import torch
from sklearn_extra.cluster import KMedoids
import logging 
import numpy as np
from random import random
logger = logging.getLogger(__name__)

def independent_cascade(g, initial_infected, p_infection, seed=None, verbose=False):
    """Simulates independent cascade model to obtain infected nodes.

    Each node that was infected during the previous timestep has a to infect a healthy node that it shares an (outgoing) edge with. 
    This step is repeated until it converges, i.e. when no more nodes have been infected in a timestep.

    Args:
        g (dgl.heterograph.DGLHeteroGraph): The graph 
        initial_infected (torch.tensor): Tensor with node indices that are initially infected. 
        p_infection (float): Probability for an infection to spread along an edge.

    Returns:
        list: List of node indices whose final status is infected.
    """
    infected = set(initial_infected.flatten())

    infectious = infected.copy()
    # Continue as long as nodes were infected in last iteration
    while(len(infectious) > 0):
        if verbose:
            logger.info(f"Infected: {len(infected)}/{g.num_nodes()}. Active: {len(infectious)}")
        next_active_nodes = set()

        for active_node in iter(infectious):
            # Infect all healthy neighbors with probability p
            nbs = g.successors(active_node)
            healthy_nbs = torch.tensor([n for n in nbs.tolist() if n not in infected])

            new_infection_mask = torch.rand(size=(len(healthy_nbs),)) < p_infection
            new_infections = healthy_nbs[new_infection_mask].long()

            # Store newly infected nodes rounds for next iteration
            next_active_nodes.update(new_infections.tolist())

            infected.update(new_infections.tolist())

        infectious = next_active_nodes

    return torch.tensor(list(infected)).long()

def independent_cascade_pythonic(g, initial_infected, p_infection):
    """Simulates independent cascade model to obtain infected nodes.
    Implementation is equal to the independent_cascade above, but slower.
    Kept for reference as it is a bit easier to extend.

    Each node that was infected during the previous timestep has a to infect a healthy node that it shares an (outgoing) edge with. 
    This step is repeated until it converges, i.e. when no more nodes have been infected in a timestep.

    Args:
        g (dgl.heterograph.DGLHeteroGraph): The graph 
        initial_infected (torch.tensor): Tensor with node indices that are initially infected. 
        p_infection (float): Probability for an infection to spread along an edge.

    Returns:
        list: List of node indices whose final status is infected.
    """

    infected_nodes = list(initial_infected)  # copy already selected nodes
    i = 0
    while i < len(infected_nodes):
        neighbours = g.successors(infected_nodes[i])
        for v in neighbours:
            if v not in infected_nodes:
                # paper describes that weights can be used for infection as well
                # but it is not used in the implementation, leaving it here for reference.
                # edge_id = g.edge_ids(T[i], v)
                # w = g.edata['weights'][edge_id]

                if random() < p_infection:
                    infected_nodes.append(v)
        i += 1
    return torch.tensor(list(infected_nodes)).long()


def perform_influence_maximization_single(g, k, p_infection, group_key, emb_key, seed=None):
    """Runs the influence maximization (independent cascade) experiment from https://arxiv.org/abs/2105.02725.

    Args:
        g (dgl.heterograph.DGLHeteroGraph): The graph 
        k (int): The number of cluster centers from k-medoids that are infected initially.
        p_infection (float): Probability for an infection to spread along an edge.
        group_key (str): Key of group labels in g.ndata. 
        emb_key (str): Key of embeddings in g.ndata which will be used to identify seeds. 
        seed (int, optional): Seed used for reproducibility.

    Returns:
        dict: Dictionary containing the results.
    """
    # Get center nodes in embedding space by applying k-medoids algorithm
    kmedoids = KMedoids(n_clusters=k).fit(g.ndata[emb_key])
    center_nodes = kmedoids.medoid_indices_.astype(int)

    # Run experiment
    infected_indices = independent_cascade(g, center_nodes, p_infection, seed=seed)

    # Compute metrics from results
    all_node_groups = g.ndata[group_key]
    infected_node_groups = g.ndata[group_key][infected_indices]

    infected_ratios = {}
    for group in g.ndata[group_key].unique():
        infected_ratios[group.item()] = float(
            sum(infected_node_groups == group) / sum(all_node_groups == group))

    infected_ratios_pct = [ratio * 100 for ratio in infected_ratios.values()]
    disparity = float(np.var(np.array(infected_ratios_pct)))

    infected_nodes_fraction_pct = (len(infected_indices) / g.num_nodes()) * 100

    results = {
        "infected_nodes_fraction": infected_nodes_fraction_pct,
        "infected_nodes": infected_indices,
        "infected_ratios_by_group": infected_ratios,
        "disparity": disparity
    }

    logger.info(f"Infected nodes fraction (%): {infected_nodes_fraction_pct}")
    logger.info("Infected nodes fraction per group (%):")

    for group in g.ndata[group_key].unique():
        logger.info(f"    Group {group.item()}:{results['infected_ratios_by_group'][group.item()] * 100}")

    logger.info(f"Disparity: {disparity}")

    return results
