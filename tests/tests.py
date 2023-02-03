import torch
import crosswalk_reproduction
import pathlib

def test_graph_reader():

    filepath = pathlib.Path(__file__).parent.parent / "data" / "immutable" / "rice" / "rice_subset.links"
    graph_1 = crosswalk_reproduction.data_provider.read_graph(str(filepath))
    print(f"num nodes: {graph_1.num_nodes()}, num_edges: {graph_1.num_edges()}")
    assert graph_1.num_nodes() == 439
    assert graph_1.num_edges() == 19320
    assert graph_1.ndata['attributes'].ndim == 2

    filepath_2 = pathlib.Path(__file__).parent.parent / "data" / "immutable" / "twitter" / "sample_4000.links"
    graph_2 = crosswalk_reproduction.data_provider.read_graph(str(filepath_2))
    print(f"num nodes: {graph_2.num_nodes()}, num_edges: {graph_2.num_edges()}")
    assert graph_2.num_nodes() == 3753
    assert graph_2.num_edges() == 13986
    assert graph_2.ndata['attributes'].ndim == 2

    print("graph_reader: passed")

def test_estimate_node_colorfulness():
    """ Tests estimate_node_colorfulness.

    """
    # Get fully connected graph with 2 groups of 2 nodes each. We assign huge weight to edges between nodes 0 and 1.
    g = crosswalk_reproduction.data_provider.synthesize_graph(node_counts=[2, 2], edge_probabilities=torch.tensor([[1,1], [1,1]]), directed=True, init_weights_strategy="uniform")

    # Test parameters
    node_idx = 0
    walk_length = 5
    walks_per_node = 100
    group_key = "groups"

    # Test 1: Test expected colorfulness with uniform transition probabilities
    # Each newly visited node should have a 50% chance of being of another group than the start node
    colorfulness = crosswalk_reproduction.preprocessor.estimate_node_colorfulness(g, node_idx, walk_length, walks_per_node, group_key, prob=None)

    # The factor (1 - 1/walk_length) comes from the start node always being part of the walk and its own group
    expected_colorfulness = 0.5 * (1 - 1/walk_length)
    assert abs(colorfulness - (expected_colorfulness)) < 0.1, "Test failed."

    # Test 2: Test expected colorfulness with weights for transition probabilities
    # The walks should (almost) always alternate between nodes 0 and 1
    prior_weight_key = "weights"
    g.edata[prior_weight_key][g.edge_ids(0,1)] = 1e9
    g.edata[prior_weight_key][g.edge_ids(1,0)] = 1e9

    colorfulness = crosswalk_reproduction.preprocessor.estimate_node_colorfulness(g, node_idx, walk_length, walks_per_node, group_key, prob=prior_weight_key)
    expected_colorfulness = 0
    assert abs(colorfulness - expected_colorfulness) < 0.1, "Test failed."

    print("estimate_node_colorfulness: passed.")

def test_get_crosswalk_weights():
    # Test: If every node's outgoing edges are initialized uniformly (in a synthetic graph), then they should sum up to 1 after crosswalk.
    # Test: For synthetic and realistic graphs, no edge should be reweighted to nan

    # Initialize different graphs
    # Create random graph
    g_synthetic = crosswalk_reproduction.data_provider.synthesize_graph(node_counts=[20, 20], edge_probabilities=torch.tensor([[0.8,0.8], [0.8,0.8]]), directed=False)
    # Create random graph with non probabilistic uniform weights
    g_synthetic_non_probabilistic = crosswalk_reproduction.data_provider.synthesize_graph(node_counts=[20, 20, 20], edge_probabilities=torch.tensor([[0.8,0.8,0.8], [0.8,0.8,0.8], [0.8,0.8,0.8]]), directed=False, init_weights_strategy="gamma")
    # Create random graph with node 0 being isolated
    g_isolated = crosswalk_reproduction.data_provider.synthesize_graph(node_counts=[1, 10], edge_probabilities=torch.tensor([[0.0 ,0.0], [0.0, 1]]), directed=False, remove_isolated=False)
    # Read rice graph
    filepath = pathlib.Path(__file__).parent.parent / "data" / "immutable" / "rice" / "rice_subset.links"
    g_rice = crosswalk_reproduction.data_provider.read_graph(str(filepath))
    # Read twitter graph
    filepath = pathlib.Path(__file__).parent.parent / "data" / "immutable" / "twitter" / "sample_4000.links"
    g_twitter = crosswalk_reproduction.data_provider.read_graph(str(filepath))

    all_graphs = [g_synthetic, g_synthetic_non_probabilistic, g_isolated, g_rice, g_twitter]

    # Test parameters
    alpha = 0.8
    p = 0.5
    walk_length = 5
    walks_per_node = 1000
    group_key = "groups"
    prior_weights_key = "weights"
    
    # Testing time!
    for i, g in enumerate(all_graphs):
        # Use prior weights for those cases where they exist
        if prior_weights_key in g.edata:
            weights = crosswalk_reproduction.preprocessor.get_crosswalk_weights(g, alpha, p, walk_length, walks_per_node, group_key, prior_weights_key=prior_weights_key)
        else:
            weights = crosswalk_reproduction.preprocessor.get_crosswalk_weights(g, alpha, p, walk_length, walks_per_node, group_key, prior_weights_key=None)

        for node in g.nodes():
            out_edges = g.out_edges(node, form="eid")
            out_weights =  weights[out_edges]
            total_outgoing_weights = out_weights.sum().item()

            # Test that no values are NaN
            assert (out_weights.isnan() == False).all(), f"Encountered nan value for node {node} with weights {out_weights}"

            # Test that crosswalk should create probabilistic edges for all non-isolated nodes
            assert len(out_edges) == 0 or abs(total_outgoing_weights - 1) < 1e-5, f"Weights are not summing up to 1 even though node {node} has at least one outgoing edge."

        print(f"get_crosswalk_weights for graph {i+1}/{len(all_graphs)}: passed")
    print(f"get_crosswalk_weights: passed")


def test_get_fairwalk_weights():
    # Test: If every node's outgoing edges are initialized uniformly (in a synthetic graph), then they should sum up to 1 after crosswalk.
    # Test: For synthetic and realistic graphs, no edge should be reweighted to nan

    # Initialize different graphs
    # Create random graph
    g_synthetic = crosswalk_reproduction.data_provider.synthesize_graph(node_counts=[20, 20], edge_probabilities=torch.tensor([[0.8,0.8], [0.8,0.8]]), directed=False)
    # Create random graph with node 0 being isolated
    g_isolated = crosswalk_reproduction.data_provider.synthesize_graph(node_counts=[1, 10], edge_probabilities=torch.tensor([[0.0 ,0.0], [0.0, 1]]), directed=False)
    # Create random graph with weights initialized as 0
    g_zero_weight = crosswalk_reproduction.data_provider.synthesize_graph(node_counts=[1, 10], edge_probabilities=torch.tensor([[0.0 ,0.0], [0.0, 1]]), directed=False, init_weights_strategy="uniform")
    g_zero_weight.edata["weights"].fill_(0)
    # Read rice graph
    filepath = pathlib.Path(__file__).parent.parent / "data" / "immutable" / "rice" / "rice_subset.links"
    g_rice = crosswalk_reproduction.data_provider.read_graph(str(filepath))
    # Read twitter graph
    filepath = pathlib.Path(__file__).parent.parent / "data" / "immutable" / "twitter" / "sample_4000.links"
    g_twitter = crosswalk_reproduction.data_provider.read_graph(str(filepath))

    all_graphs = [g_synthetic, g_isolated, g_zero_weight, g_rice, g_twitter]

    group_key = "groups"


    for i, g in enumerate(all_graphs):
        weights = crosswalk_reproduction.preprocessor.get_fairwalk_weights(g, group_key)

        for node in g.nodes():
            out_edges = g.out_edges(node, form="eid")

            # Test if all weights sum up to 1
            out_weights =  weights[out_edges]
            total_outgoing_weights = out_weights.sum().item()

            # Test that no values are NaN
            assert (out_weights.isnan() == False).all(), f"Encountered nan value for node {node} with weights {out_weights}"

            # Additionally test for the uniformly initialized graph that all weights sum up to 1
            if g == g_synthetic:
                assert len(out_edges) == 0 or abs(total_outgoing_weights - 1) < 1e-5, f"Weights are not summing up to 1 even though node {node} has at least one outgoing edge."

        print(f"get_fairwalk_weights for graph {i+1}/{len(all_graphs)}: passed")
    
    print(f"get_fairwalk_weights: passed")


if __name__ == '__main__':
    test_graph_reader()
    test_estimate_node_colorfulness()
    test_get_crosswalk_weights()
    test_get_fairwalk_weights()