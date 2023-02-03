"""
This file is a DGL implementation for weighted graphs based on the pytorch geometric package (unweighted graphs) torch_geometric.nn.Node2Vec:
<https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/node2vec.py>

Copyright (c) 2021 Matthias Fey, Jiaxuan You <matthias.fey@tu-dortmund.de, jiaxuan@cs.stanford.edu>
"""

import torch
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader
from dgl.sampling import node2vec_random_walk

import logging
logger = logging.getLogger(__name__)

class Node2Vec(torch.nn.Module):
    """The Node2Vec model from the node2vec: Scalable Feature Learning for Networks
    <https://arxiv.org/abs/1607.00653>

    Args:
        graph (DGLGraph): The DGL based graph
        embedding_dim (int): Size of the embedding
        walk_length (int): Length of the random walks 
        context_size (int): Context size from the initial node in which the random walks are considered
        walks_per_node (int): Number of walks per node
        num_negative_samples (int): Number of negative samples used for every positive sample
        p (int): (Inverse) transition probability of going back, 1 for deepwalk
        q (int): (Inverse) transition probability for going further, 1 for deepwalk
        num_workers (int): Amount of workers
        sparse (bool, optional): Create sparse embeddings or not
        weights_key (str): The name of the DGL graph edata that contains the edge weights
    """

    def __init__(self, graph, embedding_dim, walk_length, context_size, walks_per_node, num_negative_samples, p, q, weight_key):
        super().__init__()

        assert walk_length >= context_size

        self.graph = graph
        self.num_nodes = graph.num_nodes()
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples
        self.weight_key = weight_key

        self.embedding = Embedding(self.num_nodes, embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """Resets the memory."""
        self.embedding.reset_parameters()

    def embed_weights(self):
        """Returns the embedding."""
        return self.embedding.weight

    def loader(self, **kwargs):
        """Returns the pytorch dataloader."""
        return DataLoader(range(self.num_nodes), collate_fn=self.sample, **kwargs)

    def pos_sample(self, batch):
        """Returns positive random walk samples."""
        batch = batch.repeat(self.walks_per_node)
        random_walks = node2vec_random_walk(
            self.graph, batch, p=self.p, q=self.q, walk_length=self.walk_length, prob=self.weight_key)

        walks = []
        num_walks_per_random_walk = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_random_walk):
            walks.append(random_walks[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def neg_sample(self, batch):
        """Returns negative random walk samples."""
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        random_walks = torch.randint(self.num_nodes, (batch.size(0), self.walk_length))
        random_walks = torch.cat([batch.view(-1, 1), random_walks], dim=-1)

        walks = []
        num_walks_per_random_walk = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_random_walk):
            walks.append(random_walks[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def sample(self, batch):
        """Sample function used in the dataloader."""
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)

    def loss(self, pos_rw, neg_rw):
        """Computes the loss given positive and negative random walks."""
        EPS = 1e-15

        # Calculate the positive log loss
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(pos_rw.size(0), 1, self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(pos_rw.size(0), -1, self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()

        # Calculate the negative log loss
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(neg_rw.size(0), 1, self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(neg_rw.size(0), -1, self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()

        return pos_loss + neg_loss

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.embedding.weight.size(0)}, '
                f'{self.embedding.weight.size(1)})')
