from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

import logging
logger = logging.getLogger(__name__)

def perform_node_classification_single(graph, group_key, label_key, emb_key, test_size, n_neighbors):
    """Perform node classification on graph. classify each node based on its node embedding using Label Propagation.
    Label Propagation masks a subset of the nodes and then predicts these by propagating the labels for the other nodes.

    Args:
        graph (DGLGraph): The DGL based graph.
        group_key (str): Key of group labels in g.ndata. 
        label_key (str): Key of labels in g.ndata. 
        emb_key (str): Key of embeddings in g.ndata which will be used to identify seeds. 
        test_size (float): Value between 0 and 1.0 determining the fraction of the data used for testing (default: 0.5)
        n_neighbors (int): The number of neighbors taken into account by the label propagation algorithm
    
    """
    # Get the embeddings, labels (college_id 1 to 9), and sensitive group (age group)
    embeddings = graph.ndata[emb_key]

    try:
        labels = graph.ndata[label_key]
    except KeyError:
        logger.info(f"trying to get label '{label_key}' from graph but it does not exist, skipping node classification for this graph")
        empty = {
            "accuracies": [0],
            "accuracy_mean": 0,
            "disparity": 0
        }
        return empty

    groups = graph.ndata[group_key]

    # preprocess label option found in original code but not mentioned in paper
    # for idx, label in enumerate(labels):
    #     tmp = int(label)
    #     if tmp > 5:
    #         labels[idx] = 1
    #     else:
    #         labels[idx] = 0

    # Transform to numpy arrays
    X = embeddings.numpy()

    y = np.stack((labels.numpy(), groups.numpy()), axis=1)

    # Split training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)

    # Get gamma for Label Propagation
    gamma = np.mean(pairwise_distances(X))

    # Prepare scikit learn function arrays
    X_lp = np.vstack((X_train, X_test))
    y_lp = np.hstack((y_train[:, 0].squeeze(), -1*np.ones(y_test.shape[0])))

    # Train Label Propagation model
    lp = LabelPropagation(gamma = gamma, n_neighbors=n_neighbors)
    lp.fit(X_lp, y_lp)

    pred = lp.predict(X_test)

    # Get metrics per group
    group_ids = np.unique(y_test[:, 1].squeeze())
    accuracies = []
    n_samples = []
    for group in group_ids:
        y_test_group = y_test[y_test[:, 1]==group]
        y_pred_group = pred[y_test[:, 1]==group]
        accuracy_group = 100 * accuracy_score(y_test_group[:, 0].squeeze(), y_pred_group)
        logger.info(f"accuracy for group {group}: {accuracy_group}")

        n_samples.append(len(y_pred_group))
        accuracies.append(accuracy_group)

    accuracy = np.average(np.array(accuracies), weights=n_samples)
    disparity = np.var(np.array(accuracies))

    logger.info(f"Mean weighted accuracy: {accuracy}") 
    logger.info(f"Disparity: {disparity}")
    results = {
        "accuracies": accuracies,
        "accuracy_mean": accuracy,
        "disparity": disparity
    }

    return results