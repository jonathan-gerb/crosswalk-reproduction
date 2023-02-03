###########################################################
# This file contains default parameters for all the       #
# models, methods and experiments used in this repository #
###########################################################
from __future__ import absolute_import, division, print_function

from pathlib import Path

from yacs.config import CfgNode as CN

_C = CN()

###########################################################
# General configs                                         #
###########################################################
_C.SEED = 0  # from their argparser, often they dont pass it to functions
# TODO, relative path to the base directory of results
_C.BASEDIR = str(Path(__file__).parents[2] / 'results')
# path to config file used to update the defaults, do not set here!
_C.CFG_PATH = ""

# this parameter determines the task to be performed, even though parameters
# for all tasks are saved in this config, not all of them have to be performed
# options are: infmax, nodeclass, linkpred, all.
# if all is passed, all experiments will be run given the configuration
_C.TASK = 'infmax'

# TODO RESUME and STAGE argument should allow you to specify an existing experiment run and generate
# further results from earlier runs. setting the RESUME argument to an existing experiment number
# will make the main method look at the STAGE argument and try to resume the pipeline from there
# and calculate further results
_C.RESUME = -1  # ID of experiment to resume
# STAGES
# 0. perform entire experiment
# 1. Reweight graph and train node embeddings
# 2. Perform downstream task(s).
_C.STAGE = 0

# amount of repeats of the entire pipeline that will be averaged over, needs to be > 0
_C.RUNS = 1

# dataset name, should be either the dataset name or synthetic.
_C.DATASET = "synthetic"

# extra name parameter to keep track of extra single parameter changes during run all sweep
# gets added to results results.csv file as extra column, use this to distinguish
# paramter sweeps/multiple trials of the same general setup.
_C.EXTRA_LOG_NAME = "" 

_C.DATASET_PATHS = CN()
_C.DATASET_PATHS.RICE = 'data/immutable/rice/rice_subset.attr'
_C.DATASET_PATHS.RICE_RAW = 'data/immutable/rice/rice_raw.attr'  # contains the missing attributes
_C.DATASET_PATHS.TWITTER = 'data/immutable/twitter/sample_4000.attr'

_C.GRAPH_KEYS = CN()
_C.GRAPH_KEYS.GROUP_KEY = "groups"
# Key for the weights to use in case the loaded graph already has edge weights
_C.GRAPH_KEYS.PRIOR_WEIGHTS_KEY = 'weights'
# Key to use for saving the adjusted weights from the reweighting method
_C.GRAPH_KEYS.WEIGHT_KEY = 'adjusted_weights'
_C.GRAPH_KEYS.EMB_KEY = 'emb'

###########################################################
# Dataset generation configs                              #
# contains parameters for synthetic dataset to generate   #
###########################################################
_C.SYNTHETIC = CN()
# from paper, different values possible
_C.SYNTHETIC.NODES = [20, 20]
# from paper, different values possible
_C.SYNTHETIC.EDGE_PROBABILITIES = [[0.8, 0.8], [0.8, 0.8]]
_C.SYNTHETIC.SELF_CONNECT_PROB = 0.0                            # our theoretical expansion
_C.SYNTHETIC.DIRECTED = False                                   # native to our implementation
_C.SYNTHETIC.INIT_WEIGHTS_STRATEGY = "uniform"                  # native to our implementation

###########################################################
# Graph reweighting algorithm parameters                  #
# see crosswalk.py for further documentation              #
###########################################################
_C.GRAPH_WEIGHTING = CN()
# from paper, can take many values
_C.GRAPH_WEIGHTING.ALPHA = 0.5
# from paper, can take many values
_C.GRAPH_WEIGHTING.P = 4.0
# got this from their bash file, but also 10 and 20 are there commented out
_C.GRAPH_WEIGHTING.WALK_LENGTH = 5
# this is a hard coded value for them
_C.GRAPH_WEIGHTING.WALKS_PER_NODE = 1000
# options are ['crosswalk', 'fairwalk', 'skip']
_C.GRAPH_WEIGHTING.METHOD = 'crosswalk'
# True for the original version, False for the version proposed during the 2023 reproducibility study
_C.GRAPH_WEIGHTING.USE_ORIGINAL_CROSSWALK_IMPLEMENTATION = False

###########################################################
# Node embedding training parameters                      #
###########################################################
_C.EMBEDDINGS = CN()

_C.EMBEDDINGS.USE_GENSIM = True             # use original gensim word2vec impl
_C.EMBEDDINGS.MIN_COUNT = 0                 # from original paper, gensim option
_C.EMBEDDINGS.METHOD = 'node2vec'           # options are ['node2vec', 'deepwalk']
_C.EMBEDDINGS.EMBEDDING_DIM = 32            # from their bash file
_C.EMBEDDINGS.WALK_LENGTH = 40              # from their bash file
_C.EMBEDDINGS.CONTEXT_SIZE = 10             # from their bash file
_C.EMBEDDINGS.WALKS_PER_NODE = 80           # from their bash file, also 160 is there commented out
_C.EMBEDDINGS.NUM_NEGATIVE_SAMPLES = 5      # standard gensim value
_C.EMBEDDINGS.P = 1.0                       # this is for deepwalk
_C.EMBEDDINGS.Q = 1.0                       # this is for deepwalk
_C.EMBEDDINGS.NUM_WORKERS = 6               # they had 30 but most pc's do not like that so use 4
_C.EMBEDDINGS.BATCH_SIZE = 32               # picked by us
_C.EMBEDDINGS.LR = 0.025                    # from gensim word2vec defaults
_C.EMBEDDINGS.NUM_EPOCHS = 5                # picked by us
# gensim implementation uses sgd+momentum which is the other option using 'sgd'
_C.EMBEDDINGS.OPTIMIZER = 'adam'
_C.EMBEDDINGS.DISABLE_PBAR = None           # native to our implementation
_C.EMBEDDINGS.EMBEDDING_ID = -1             # native to our implementation, id for the embedding to load

###########################################################
# Downstream task parameters                              #
###########################################################

# influence maximization parameters
# random values, get nicer defaults please
_C.INFLUENCE_MAXIMIZATION = CN()
# 0.01 for real datasets and 0.03 for synthetic datasets from paper
_C.INFLUENCE_MAXIMIZATION.P_INFECTION = 0.01
_C.INFLUENCE_MAXIMIZATION.K = 40                # from paper

# link prediction parameters
_C.LINK_PREDICTION = CN()
_C.LINK_PREDICTION.TEST_SIZE = 0.1              # from paper

# node classification parameters
_C.NODE_CLASSIFICATION = CN()
_C.NODE_CLASSIFICATION.TEST_SIZE = 0.5          # from paper
_C.NODE_CLASSIFICATION.N_NEIGHBORS = 7          # from paper

# cfgs that are set in setup code
_C.EXPERIMENT_PATH = ""


def str_to_pathlib(cfg):
    """Convert strings to pathlib paths and back.
    This ensures that paths work on all operating systems.
    Add new arguments with filepaths here as well.

    Args:
        cfg (yacs.config.CfgNode): The original config.
    """
    cfg.BASEDIR = str(Path(cfg.BASEDIR))
    cfg.DATASET_PATHS.RICE = str(Path(cfg.DATASET_PATHS.RICE))
    cfg.DATASET_PATHS.TWITTER = str(Path(cfg.DATASET_PATHS.TWITTER))

    return cfg


def verify_config(cfg):
    """Verifies the options given for the hyperparameters.

    Args:
        cfg (yacs.config.CfgNode): config to check

    Returns:
        yacs.config.CfgNode: verified config.
    """
    cfg.DATASET = cfg.DATASET.lower()
    cfg.TASK = cfg.TASK.lower()
    assert cfg.DATASET in ['rice', 'twitter', 'synthetic']
    assert cfg.TASK in ['linkpred', 'infmax', 'nodeclass', 'all']
    assert cfg.GRAPH_WEIGHTING.METHOD in ['crosswalk', 'fairwalk', 'skip']
    assert cfg.RUNS > 0


def update_config(cfg, args, freeze=True):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg = str_to_pathlib(cfg)

    # verify config
    verify_config(cfg)

    if freeze:
        cfg.freeze()
    return cfg


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
