from pathlib import Path

import dgl
import numpy as np
import torch
from dgl.nn import EdgeWeightNorm

from ..graphs import get_uniform_weights

import logging
logger = logging.getLogger(__name__)


def read_attr_file(attrib_filepath):
    """Read .attr graph file.

    Args:
        attrib_filepath (pathlib.Path): Pathlib path to file, can be .links
        file as well as long as the rest of the path is the same.

    Returns:
        dict: dict containing the attributes per node
    """
    attributes = {}
    with open(str(attrib_filepath.with_suffix(".attr"))) as f:
        for node in f:
            line = node.strip().split()
            id = line[0]
            attr_list = []
            for idx in range(1, len(line)):
                if "." in line[idx]:
                    attr = float(line[idx])
                else:
                    attr = int(line[idx])
                attr_list.append(attr)

            attributes[int(id)] = torch.Tensor(attr_list)
    return attributes

def read_links_file(link_filepath):
    """Read .link graph edge file.

    Args:
        link_filepath (pathlib.Path): Pathlib path to the file, can be .attr

    Raises:
        NotImplementedError: if attribute file contains more than 1 edge attribute

    Returns:
        list: list containing a dict for each edge. 
    """
    links = []
    # read edges
    with open(str(link_filepath.with_suffix(".links"))) as f:
        for link in f:
            link = link.replace("[", "")
            link = link.replace("]", "")
            link = link.replace(",", "")
            linkdata = link.strip().split()

            # unweighted graph
            if len(linkdata) == 2:
                link_dict = {
                    "source_id": int(linkdata[0]),
                    "target_id": int(linkdata[1]),
                    "edge_attributes": 1.0  # default edge attribute is 1.0
                }
                links.append(link_dict)

            # weighted undirected graph
            elif len(linkdata) == 3:
                link_dict = {
                    "source_id": int(linkdata[0]),
                    "target_id": int(linkdata[1]),
                    "edge_attributes": np.array([float(linkdata[2])])
                }
                links.append(link_dict)

            # weighted directed graph
            elif len(linkdata) == 4:
                # first add edge in one direction.
                link_dict = {
                    "source_id": int(linkdata[0]),
                    "target_id": int(linkdata[1]),
                    "edge_attributes": np.array([float(linkdata[2])])
                }
                links.append(link_dict)
                # add edge in second direction
                link_dict = {
                    "source_id": int(linkdata[1]),
                    "target_id": int(linkdata[0]),
                    "edge_attributes": np.array([float(linkdata[3])])
                }
                links.append(link_dict)
            else:
                raise NotImplementedError(
                    f"using {len(linkdata) - 2} edge attributes is not supported")
    return links

def read_attr_links_graphs(link_filepath):
    """read graph from .attr and .links file

    Args:
        link_filepath (str): path to .attr of .links file

    Returns:
        list: list of dicts where each dict contains all edge information
        dict: dict of dicts attributes where each dict contains node information, node_index is key.
    """
    link_filepath = Path(link_filepath)
    # read node attributes
    attributes = read_attr_file(link_filepath)
    # read edges  attributes
    links = read_links_file(link_filepath)

    return links, attributes


def graph_from_la(links, attributes, weight_key="weights", group_key="groups"):
    """Construct graph from links and attributes files. 

    Args:
        links (list): list of dicts where each dict contains all edge information 
        attributes (dict): dict of dicts attributes where each dict contains node information, node_index is key.

    Returns:
        torch_geometric.data.Data: Graph constructed from input
    """
    # get all id's that are in links
    src_ids = [edgedata['source_id'] for edgedata in links]
    target_ids = [edgedata['target_id'] for edgedata in links]
    all_link_ids = list(set(src_ids + target_ids))

    edge_weight_by_src_target = {(link_dict['source_id'], link_dict['target_id']): link_dict['edge_attributes']  for link_dict in links}
    edge_tuples = [(src, target) for (src, target) in zip(src_ids, target_ids)]
    
    edge_weights = [edge_weight_by_src_target[edge_tuple] for edge_tuple in edge_tuples]
    edge_weights = torch.Tensor(np.array(edge_weights)).type(torch.float32)

    # all id's that have an attribute associated with them
    all_attribute_ids = list(set(list(attributes.keys())))

    # all id's in the dataset
    all_found_ids = list(set(all_link_ids + all_attribute_ids))

    # get all id's that we have attributes for
    only_attributed_ids = list(set([id for id in all_found_ids if id in all_attribute_ids]))

    only_attributed_linked_ids = [id for id in only_attributed_ids if id in all_link_ids]

    # all links between nodes that we have attributes for
    only_attributed_links = list(set([(src_id, target_id) for src_id, target_id in zip(src_ids, target_ids) 
                             if src_id in only_attributed_linked_ids and target_id in only_attributed_linked_ids]))
    
    logger.info(f"number of attributed ids: {len(only_attributed_ids)}, ids with edges: {len(only_attributed_linked_ids)}, number of links with attributes: {len(only_attributed_links)}")

    new_node_ids = list(range(len(only_attributed_linked_ids)))

    old_to_new_nodeid = {old_node: new_node for old_node, new_node in zip(only_attributed_linked_ids, new_node_ids)}

    # replace all node ids
    src_ids = [old_to_new_nodeid.get(edgedata[0]) for edgedata in only_attributed_links]
    target_ids = [old_to_new_nodeid.get(edgedata[1]) for edgedata in only_attributed_links]

    # make tensors
    src_ids_tensor = torch.Tensor(np.array(src_ids)).int()
    target_ids_tensor = torch.Tensor(np.array(target_ids)).int()

    # get new node id's for attributes
    old_ids = torch.tensor([k for k, v in attributes.items() if k in old_to_new_nodeid])
    new_ids = torch.tensor([old_to_new_nodeid[k] for k, v in attributes.items() if k in old_to_new_nodeid])

    attributes_tensor = torch.stack([v for k, v in attributes.items()
                                  if k in old_to_new_nodeid])

    missing_ids = torch.tensor([k for k, v in attributes.items() if k not in old_to_new_nodeid])
    if len(missing_ids) > 0:
        logger.info(
            f"Found unconnected nodes in graph! Removing {len(missing_ids.tolist())} unconnected nodes")

    # construct graph
    graph = dgl.graph((src_ids_tensor, target_ids_tensor), num_nodes=len(new_node_ids))

    # add node and edge data
    # all node data, first index is likely the protected attribute
    graph.ndata['attributes'] = attributes_tensor.type(torch.int64)

    # add new node id and original node id
    graph.ndata['id'] = new_ids.type(torch.int64)
    graph.ndata['id_original'] = old_ids.type(torch.int64)

    # use first attribute for grouping. Overwrite to correct group later
    graph.ndata[group_key] = attributes_tensor[:, 0].type(torch.int64)

    if torch.all(edge_weights == 1.0):
        logger.info("no edge weights given, initializing all as uniform normalized.")
        # norm = EdgeWeightNorm(norm='left')
        # uni_weights = norm(graph, torch.ones_like(edge_weights))

        # custom method
        uni_weights = get_uniform_weights(graph)
        
        graph.edata[weight_key] = uni_weights
    else:
        logger.info("using weights from file.")
        graph.edata[weight_key] = edge_weights.reshape(-1)

    # by default all int values are int64
    graph = graph.long()

    # ToSimple removes parallel edges.
    transform = dgl.transforms.ToSimple()
    graph = transform(graph)

    # sort nodes of the graph by new id, you should not get nodes
    # by index ever, but if it happens somewhere this should prevent 
    # that from giving errors. ie. ndata[key][50] will give the node data for node with id 50
    # better way to get that would be g.ndata[key][g.ndata['id'] == 50]

    _, sorted_idx = graph.ndata['id'].sort()
    for key in graph.ndata.keys():
        graph.ndata[key] = graph.ndata[key][sorted_idx]

    return graph


def set_group_rice(graph, group_attribute, group_key):
    """parse rice dataset to set the 'groups' node label from one of the attributes from the raw file

    Args:
        graph (dgl.heterograph.DGLHeteroGraph): graph to modify
        group_attribute (str): group attribute to use as label, must be 'college', 'age', or 'major'

    Returns:
        dgl.heterograph.DGLHeteroGraph: modified graph
    """
    if group_attribute == None:
        # logger.info("no group attribute given, using idx 1: 'age'")
        group_attribute = "age"

    attribute_keys = {
        "college": 0,
        "age": 1,
        "major": 2 
    }
    if group_attribute not in attribute_keys:
        raise ValueError(f"no attribute {group_attribute} in rice dataset.\n chose from {list(attribute_keys.keys())}")

    idx = attribute_keys[group_attribute]
    graph.ndata[group_key] = graph.ndata['attributes'][:, idx]

    graph.ndata["college"] = graph.ndata['attributes'][:, 0]
    graph.ndata["age"] = graph.ndata['attributes'][:, 1]
    graph.ndata["major"] = graph.ndata['attributes'][:, 2]

    # construct groups from sensitive attribute data
    group_1_idx = graph.ndata[group_key] == 20
    group_2_idx = (graph.ndata[group_key] == 19) | (graph.ndata[group_key] == 18)
    group_3_idx = ~((group_1_idx) | (group_2_idx))

    # assign group numbers
    graph.ndata[group_key][group_1_idx] = 0
    graph.ndata[group_key][group_2_idx] = 1

    # remove nodes that are not in the right age range
    idx = np.arange(graph.num_nodes())
    nodes_to_remove = idx[group_3_idx]
    graph = dgl.remove_nodes(graph, nodes_to_remove)

    return graph


def add_raw_attributes(graph, filepath):
    """add missing attributes from rice_raw data to rice_subset dataset

    Args:
        graph (dgl.heterograph.DGLHeteroGraph): graph to add attributes to, needs to have 'id_original' key
        filepath (str): filepath to the rice_raw.attr file

    Returns:
        dgl.heterograph.DGLHeteroGraph: graph with the missing attributes added to ndata['attributes']
    """
    attributes = read_attr_file(filepath)
    n_attributes = len(list(attributes.values())[0])

    new_attribs = torch.zeros(graph.ndata['id_original'].shape[0], n_attributes) -1

    for idx, original_id in enumerate(graph.ndata['id_original']):
        new_attribs[idx] = attributes[int(original_id)]

    graph.ndata['attributes'] = new_attribs
    return graph


def read_graph(filepath, weight_key="weights", group_key="groups", group_attribute=None):
    """Read graph from filepath

    Accepts multiple types of input file structures

    Args:
        filepath (str): path to graph file

    Returns:
        _type_: _description_
    """
    filepath = Path(filepath)
    # read graph like used in the original crosswalk repository
    if filepath.suffix == ".attr" or filepath.suffix == ".links":
        links, attributes = read_attr_links_graphs(filepath.with_suffix(".links"))
        graph = graph_from_la(links, attributes, weight_key=weight_key, group_key=group_key)

        if "rice" in str(filepath):
            graph = add_raw_attributes(graph, Path(filepath).with_name('rice_raw.attr'))
            logger.info("detected rice_raw dataset!")
            logger.info(f"before rice parsing! num nodes:{graph.num_nodes()}, num edges:{graph.num_edges()}")
            graph = set_group_rice(graph, group_attribute, group_key)
            logger.info(f"after rice parsing! num nodes:{graph.num_nodes()}, num edges:{graph.num_edges()}")

    return graph
