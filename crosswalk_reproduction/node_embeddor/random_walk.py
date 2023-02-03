import logging
import sys

import numpy as np
import torch
from crosswalk_reproduction.node_embeddor.models import DeepWalk, Node2Vec
from dgl.sampling import node2vec_random_walk, random_walk
from tqdm import tqdm

logger = logging.getLogger(__name__)


def generate_embeddings(graph, **kwargs):
    """Generate the node embeddings for given graph using random walk algorithm and write to disk.

    This method performs the training of node embeddings.

    Args:
        graph (DGLGraph): The DGL based graph with each node having at least one outgoing edge.
        method (str): The random walk based node representation learning algorithm used 
        gensim (bool): Use gensim to get the embeddings, this method was used in the crosswalk paper.
        seed (int): The random seed used for reproducability
        embedding_dim (int): Size of the embedding
        walk_length (int): Length of the random walks 
        context_size (int): Context size from the initial node in which the random walks are considered
        walks_per_node (int): Number of walks per node
        num_negative_samples (int): Number of negative samples used for every positive sample
        p (int): (Inverse) transition probability of going back, 1 for deepwalk
        q (int): (Inverse) transition probability for going further, 1 for deepwalk
        num_workers (int): Amount of workers
        batch_size (int): Batch size of amount of random walks 
        lr (float): Learning rate
        disable_pbar (bool): If running on a server you might want to disable the progress bar
        weights_key (str): The name of the DGL graph edata that contains the edge weights
    """

    assert (graph.out_degrees() != 0).all(
    ), "This method expects all nodes to have at least one outgoing edge."

    # Get kwargs, same as what's passed in add_embeddings
    use_gensim = kwargs.get("use_gensim", True)
    method = kwargs.get("method", "Node2Vec")
    embedding_dim = kwargs.get("embedding_dim", 32)
    walk_length = kwargs.get("walk_length", 40)
    context_size = kwargs.get("context_size", 10)
    walks_per_node = kwargs.get("walks_per_node", 80)
    num_negative_samples = kwargs.get("num_negative_samples", 5)
    p = kwargs.get("p", 1.0)
    q = kwargs.get("q", 1.0)
    num_workers = kwargs.get("num_workers", 4)
    batch_size = kwargs.get("batch_size", 128)
    lr = kwargs.get("lr", 0.025)
    num_epochs = kwargs.get("num_epochs", 10)
    disable_pbar = kwargs.get("disable_pbar", None)
    weight_key = kwargs.get("weight_key", "adjusted_weights")
    optimizer = kwargs.get("optimizer", "adam")
    min_count = kwargs.get("min_count", 0)

    if use_gensim:
        if method == "node2vec":
            skipgram = 1
            hierarchical_softmax = 0
        elif method == 'deepwalk':
            skipgram = 1
            hierarchical_softmax = 1
        else:
            raise NotImplementedError(f"no node embedding method: {method} implemented")

        try:
            from gensim.models import Word2Vec
        except Exception as e:
            logging.error(
                f"{e}\n Gensim was not installed, please install gensim to generate the word embeddings")
            sys.exit(0)

        logger.info("generating embeddings using gensim")
        start_nodes = (graph.nodes().repeat(walks_per_node)).type(graph.idtype).to('cpu')

        if method == "node2vec":
            random_walks = node2vec_random_walk(
                graph, start_nodes, walk_length=walk_length, prob="adjusted_weights", p=p, q=q)
        elif method == 'deepwalk':
            random_walks, _ = random_walk(
                graph, start_nodes, length=walk_length, prob="adjusted_weights")

        # Word2Vec needs node id's to be strings otherwise it will give you a
        # VERY descriptive and helpful error..
        random_walks = np.array(random_walks.tolist()).astype(str).tolist()
        model = Word2Vec(alpha=lr, sentences=random_walks, vector_size=embedding_dim, window=context_size,
                         min_count=min_count, workers=num_workers, sg=skipgram, hs=hierarchical_softmax,
                         negative=num_negative_samples, epochs=num_epochs)

        embedding_lst = list()

        for node in graph.nodes():
            node_embedding = list(model.wv[f"{node.item()}"].astype(float))
            embedding_lst.append(node_embedding)

        embedding = torch.Tensor(embedding_lst)
        return embedding
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"training on device: {device}")
        # Get random walk based node representation learning algorithm to use
        if method == "deepwalk":
            model = DeepWalk(graph, embedding_dim=embedding_dim, walk_length=walk_length,
                             context_size=context_size, walks_per_node=walks_per_node,
                             num_negative_samples=num_negative_samples,
                             weight_key=weight_key).to(device)
        elif method == "node2vec":
            model = Node2Vec(graph, embedding_dim=embedding_dim, walk_length=walk_length,
                             context_size=context_size, walks_per_node=walks_per_node,
                             num_negative_samples=num_negative_samples, p=p, q=q,
                             weight_key=weight_key).to(device)
        else:
            raise NotImplementedError(f"no node embedding method: {method} implemented")

        loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=num_workers)

        logging.info(
            f"training {method} model with : {optimizer}, lr={lr}, batch_size={batch_size}")
        if optimizer == "adam":
            optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)
        elif optimizer == 'sgd':
            optimizer = torch.optim.SGD(list(model.parameters()), lr=lr, momentum=0.9)
        else:
            raise NotImplementedError(f"no optimizer: {optimizer} Implemented")

        model.train()
        tbar_1 = tqdm(range(num_epochs), desc=" Epoch", disable=disable_pbar)

        epoch_losses = 0
        for epoch in tbar_1:
            tbar_1.set_description(f"epoch ({epoch+1}) loss: {epoch_losses}")
            tbar_2 = tqdm(loader, desc=" iteration", disable=disable_pbar, leave=None)

            epoch_loss = []
            for pos_samples, neg_samples in tbar_2:
                optimizer.zero_grad()
                loss = model.loss(pos_samples.to(device), neg_samples.to(device))
                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.item())
                tbar_2.set_description(f"mean running iteration loss: {np.mean(epoch_loss[-10:])}")

            epoch_losses = np.mean(epoch_loss)
        tbar_1.close()

        embedding = model.embed_weights().detach().cpu()

        return embedding
