import logging

import dgl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def get_edge_types(g, group_key, same_id_for_both_directions = True):
    """Creates edge types defined by the group of the nodes the edge connects.

    Args:
        g (dgl.heterograph.DGLHeteroGraph): The graph.
        group_key (str): Key of group labels in g.ndata.
        same_id_for_both_directions (bool, optional): Whether the edge type of (g1,g2) should be the same as that of (g2,g1).
            Should be True for undirected graphs.
            Defaults to True.

    Returns:
        dict of tuple of groups to int: Dictionary containing edge type ids with keys being tuples of groups.
    """
    groups = g.ndata[group_key].unique().tolist()
    edge_types = {}

    # set id for all edge types with group_1 >= group_2
    for i, group_1 in enumerate(groups):
        for j, group_2 in enumerate(groups):
            if i >= j:
                edge_types[(group_1, group_2)] = len(edge_types)

    # set id for all edge types with group_1 < group_2
    for i, group_1 in enumerate(groups):
        for j, group_2 in enumerate(groups):
            if i < j:
                # If specified, use ID that was already created in first run
                if same_id_for_both_directions:
                    edge_types[(group_1, group_2)] = edge_types[(group_2, group_1)]
                # Otherwise create new ID
                else:
                    edge_types[(group_1, group_2)] = len(edge_types)

    return edge_types

def perform_link_prediction_single(graph, group_key, test_size=0.1, seed=42):
    """Do link prediction with Logistic Regression.

    Args:
        graph (DGLGraph): The DGL based graph.
        group_key (str): Key of group labels in g.ndata. Used to create targets for edges.
        seed (int): The random seed used for reproducability (default: 42)
        test_size (float, optional): Value between 0 and 1.0 determining the fraction of the data used for testing (default: 0.1)

    Returns:
        dict: Dictionary containing the results.
    """

    edge_type_dict = get_edge_types(graph, group_key)

    # Get positive samples
    left_edge_pos, right_edge_pos = graph.edges()
    x_pos = list() # list of edge features
    y_pos = list() # list of (exists, edge_type)
    for i in range(graph.num_edges()):
        left_node, right_node = left_edge_pos[i].item(), right_edge_pos[i].item()

        feature = (graph.ndata["emb"][left_node] - graph.ndata["emb"][right_node])**2
        x_pos.append(feature.numpy().astype(float))

        left_node_group, right_node_group = graph.ndata["groups"][left_node].item(), graph.ndata["groups"][right_node].item()
        edge_type = edge_type_dict[(left_node_group, right_node_group)]
        y_pos.append(np.array([1, edge_type]))


    # Add as many negative samples as real edges exist
    left_edge_neg, right_edge_neg = dgl.sampling.global_uniform_negative_sampling(graph, graph.num_edges())
    x_neg = list() # list of edge features
    y_neg = list() # list of (exists, edge_type)
    for i in range(left_edge_neg.shape[0]):
        left_node, right_node = left_edge_neg[i].item(), right_edge_neg[i].item()

        feature = (graph.ndata["emb"][left_node] - graph.ndata["emb"][right_node])**2
        x_neg.append(feature.numpy().astype(float))

        left_node_group, right_node_group = graph.ndata["groups"][left_node].item(), graph.ndata["groups"][right_node].item()
        edge_type = edge_type_dict[(left_node_group, right_node_group)]
        y_neg.append(np.array([0, edge_type]))

    # Split the data into training data and test data
    # Taking the same amount of positive and negative samples for each edge type
    x_pos_train, x_pos_test, y_pos_train, y_pos_test = train_test_split(np.array(x_pos), np.array(y_pos), test_size=test_size, random_state=seed, shuffle=True, stratify=np.array(y_pos))
    x_neg_train, x_neg_test, y_neg_train, y_neg_test = train_test_split(np.array(x_neg), np.array(y_neg), test_size=test_size, random_state=seed, shuffle=True, stratify=np.array(y_neg))

    x_train = np.concatenate((x_pos_train, x_neg_train))
    x_test = np.concatenate((x_pos_test, x_neg_test))
    y_train = np.concatenate((y_pos_train, y_neg_train))
    y_test = np.concatenate((y_pos_test, y_neg_test))


    # Train Logistic regression
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(x_train, y_train[:, 0].squeeze())

    # Get metrics per group
    edge_types = np.unique(y_test[:, 1].squeeze())
    y_pred = lr.predict(x_test)

    accuracies = []
    n_samples = []
    for edge_type in edge_types:
        group_nodes = y_test[:, 1] == edge_type
        y_test_group = y_test[group_nodes]
        y_pred_group = y_pred[group_nodes]

        accuracy_group = accuracy_score(y_test_group[:, 0].squeeze(), y_pred_group)

        accuracies.append(accuracy_group * 100)
        n_samples.append(len(y_pred_group))
        logger.info(f"Edge type {edge_type} has accuracy: {accuracy_group * 100}")

    accuracy = accuracy_score(y_test[:, 0].squeeze(), y_pred) * 100
    disparity = np.var(np.array(accuracies))

    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"Disparity: {disparity}")
    results = {'accuracy': float(accuracy), 'disparity': float(disparity)}

    return results
