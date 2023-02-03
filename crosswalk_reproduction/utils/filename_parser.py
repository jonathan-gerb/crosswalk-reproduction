import glob
import os
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

def generate_filename(cfg):
    """Generate filename from arguments.

    Returns:
        str: filename
    """
    # get unused experiment number
    experiments = glob.glob(str(cfg.BASEDIR) + "/experiment_*")
    numbers = [int(os.path.basename(fp).split("_")[1]) for fp in experiments]
    if len(numbers) == 0:
        i = 0
    else:
        i = max(numbers) + 1

    if "synthetic" in cfg.DATASET:
        filename = f"experiment_{i}_{cfg.TASK}_{cfg.DATASET}{len(cfg.SYNTHETIC.NODES)}_{cfg.EMBEDDINGS.METHOD}_{cfg.GRAPH_WEIGHTING.METHOD}"
    else:
        filename = f"experiment_{i}_{cfg.TASK}_{cfg.DATASET}_{cfg.EMBEDDINGS.METHOD}_{cfg.GRAPH_WEIGHTING.METHOD}"

    return filename


def generate_emb_filepath(cfg):
    experiments = glob.glob(str(Path(cfg.EXPERIMENT_PATH) / "embeddings") + "/*")
    numbers = [int(os.path.basename(fp).split("_")[1]) for fp in experiments]
    if len(numbers) == 0:
        i = 0
    else:
        i = max(numbers) + 1

    emb_filename = f"emb_{i}_{cfg.EMBEDDINGS.METHOD}_" + \
        f"ed{cfg.EMBEDDINGS.EMBEDDING_DIM}_" + \
        f"wl{cfg.EMBEDDINGS.WALK_LENGTH}_" + \
        f"cs{cfg.EMBEDDINGS.CONTEXT_SIZE}_" + \
        f"wpn{cfg.EMBEDDINGS.WALKS_PER_NODE}_" + \
        f"nns{cfg.EMBEDDINGS.NUM_NEGATIVE_SAMPLES}.pt"

    emb_filepath = Path(cfg.EXPERIMENT_PATH) / "embeddings" / emb_filename
    return emb_filepath


def get_experiment_folder(basedir, experiment_id):
    experiment_folder = glob.glob(str(basedir) + f"/experiment_{experiment_id}*")[0]
    return experiment_folder


def get_embedding_file(exp_dir, embedding_id):
    try:
        experiment_folder = glob.glob(
            str(Path(exp_dir) / 'embeddings') + f"/emb_{embedding_id}*.pt")[0]
    except IndexError:
        raise FileNotFoundError(
            f"could not find embedding with ID: {embedding_id} at directory {exp_dir}/embeddings")

    return experiment_folder