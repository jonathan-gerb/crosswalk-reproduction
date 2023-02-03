import argparse
import datetime
import glob
import logging
import os
from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import torch
from dgl.data.utils import load_graphs, save_graphs

from .config import get_cfg_defaults, update_config
from .data_provider import read_graph, synthesize_graph
from .node_embeddor import generate_embeddings
from .preprocessor import get_crosswalk_weights, get_fairwalk_weights
from .tasks import (perform_influence_maximization_single,
                    perform_link_prediction_single,
                    perform_node_classification_single)
from .utils import filename_parser

logging.getLogger('gensim').setLevel(logging.WARNING)
logger = logging.getLogger()


def set_seed(seed):
    """Set the pytorch seed.

    Args:
        seed (int): The random seed used for reproducability
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_or_construct_graph(cfg):
    graph_filepath = str(Path(cfg.EXPERIMENT_PATH) / "graph" / "graph.bin")
    # if not resuming
    if cfg.RESUME == -1:
        if cfg.DATASET.lower() == "synthetic":
            logger.info("Generating new graph..")
            graph = synthesize_graph(cfg.SYNTHETIC.NODES, torch.Tensor(cfg.SYNTHETIC.EDGE_PROBABILITIES),
                                     self_connection_prob=cfg.SYNTHETIC.SELF_CONNECT_PROB, directed=cfg.SYNTHETIC.DIRECTED,
                                     init_weights_strategy=cfg.SYNTHETIC.INIT_WEIGHTS_STRATEGY, weight_key=cfg.GRAPH_KEYS.PRIOR_WEIGHTS_KEY, group_key=cfg.GRAPH_KEYS.GROUP_KEY)
        else:
            if cfg.DATASET.lower() == 'rice':
                logger.info(f"reading: {cfg.DATASET_PATHS.RICE}")
                graph = read_graph(cfg.DATASET_PATHS.RICE,
                                   weight_key=cfg.GRAPH_KEYS.PRIOR_WEIGHTS_KEY,
                                   group_key=cfg.GRAPH_KEYS.GROUP_KEY)
            elif cfg.DATASET.lower() == 'twitter':
                graph = read_graph(cfg.DATASET_PATHS.TWITTER,
                                   weight_key=cfg.GRAPH_KEYS.PRIOR_WEIGHTS_KEY,
                                   group_key=cfg.GRAPH_KEYS.GROUP_KEY)
            else:
                raise NotImplementedError(f"no known dataset: {cfg.DATASET}")
            logger.info(f"loaded graph: {cfg.DATASET.lower()}")

        # save graph
        save_graphs(graph_filepath, [graph], labels=None)
        logger.info("saved graph")
    # if resuming
    else:
        # load graph,
        # we assume each graph.bin file only contains one graph, hence the [0][0]
        graph = load_graphs(graph_filepath)[0][0]
        logger.info(f"loaded graph from: '{graph_filepath}'")

    return graph


def reweight_edges(cfg, graph):
    """Apply one of the edge reweighting methods to increase fairness for random walk based node embeddings.add()

    Args:
        cfg (yacs.config.CfgNode): cfg for the experiment
        graph (dgl.heterograph.DGLHeteroGraph): graph with weights in the graph.edata[cfg.GRAPH_KEYS.PRIOR_WEIGHTS_KEY]

    Returns:
        dgl.heterograph.DGLHeteroGraph: graph with reweighted edges at graph.edata[cfg.GRAPH_KEYS.WEIGHTS_KEY]
    """
    adjusted_weights = None
    # if resuming at stage 1 or not resuming, reweight graph
    if (cfg.RESUME > -1 and cfg.STAGE == 1) or cfg.RESUME == -1:
        if cfg.GRAPH_WEIGHTING.METHOD == 'crosswalk':
            logger.info("performing edge reweighting with crosswalk")
            # reweighting graph
            adjusted_weights = get_crosswalk_weights(graph,
                                                     cfg.GRAPH_WEIGHTING.ALPHA,
                                                     cfg.GRAPH_WEIGHTING.P,
                                                     cfg.GRAPH_WEIGHTING.WALK_LENGTH,
                                                     cfg.GRAPH_WEIGHTING.WALKS_PER_NODE,
                                                     cfg.GRAPH_KEYS.GROUP_KEY,
                                                     cfg.GRAPH_KEYS.PRIOR_WEIGHTS_KEY,
                                                     use_original_crosswalk_implementation=cfg.GRAPH_WEIGHTING.USE_ORIGINAL_CROSSWALK_IMPLEMENTATION
                                                     )

        elif cfg.GRAPH_WEIGHTING.METHOD == 'fairwalk':
            logger.info("performing edge reweighting with fairwalk")
            adjusted_weights = get_fairwalk_weights(graph, cfg.GRAPH_KEYS.GROUP_KEY)

        elif cfg.GRAPH_WEIGHTING.METHOD == 'skip':
            logger.info("using prior edge weights without reweighting..")
            adjusted_weights = graph.edata[cfg.GRAPH_KEYS.PRIOR_WEIGHTS_KEY]
        else:
            logger.info(f"reweighting method '{cfg.GRAPH_WEIGHTING.METHOD}' not implemented")
    # if resuming but at stage 2
    else:
        logger.info("Skipping graph reweighting for stage 0, as existing node embeddings will be used.")
        # load prior weights as new weights
        adjusted_weights = graph.edata[cfg.GRAPH_KEYS.PRIOR_WEIGHTS_KEY]

    # make sure there are no nans
    assert not torch.isnan(adjusted_weights).any()

    # assign new weights to graph edges
    graph.edata[cfg.GRAPH_KEYS.WEIGHT_KEY] = adjusted_weights
    return graph


def add_embeddings(cfg, graph):
    # arguments for generating node embeddings
    embedding_args = {
        "method": cfg.EMBEDDINGS.METHOD,
        "embedding_dim": cfg.EMBEDDINGS.EMBEDDING_DIM,
        "walk_length": cfg.EMBEDDINGS.WALK_LENGTH,
        "context_size": cfg.EMBEDDINGS.CONTEXT_SIZE,
        "walks_per_node": cfg.EMBEDDINGS.WALKS_PER_NODE,
        "num_negative_samples": cfg.EMBEDDINGS.NUM_NEGATIVE_SAMPLES,
        "p": cfg.EMBEDDINGS.P,
        "q": cfg.EMBEDDINGS.Q,
        "num_workers": cfg.EMBEDDINGS.NUM_WORKERS,
        "batch_size": cfg.EMBEDDINGS.BATCH_SIZE,
        "lr": cfg.EMBEDDINGS.LR,
        "num_epochs": cfg.EMBEDDINGS.NUM_EPOCHS,
        "disable_pbar": cfg.EMBEDDINGS.DISABLE_PBAR,
        "optimizer": cfg.EMBEDDINGS.OPTIMIZER,
        "weight_key": cfg.GRAPH_KEYS.WEIGHT_KEY,
        "use_gensim": cfg.EMBEDDINGS.USE_GENSIM,
        "min_count": cfg.EMBEDDINGS.MIN_COUNT,
        "seed": cfg.SEED
    }

    # if generating new embedding
    if (cfg.RESUME > -1 and cfg.STAGE == 1) or cfg.RESUME == -1:
        logger.info("generating node embeddings..")
        embedding = generate_embeddings(graph, **embedding_args)
        emb_filepath = filename_parser.generate_emb_filepath(cfg)
        logger.info(f"saving embedding file to {emb_filepath}")
        torch.save({"embedding": embedding, "config": cfg}, emb_filepath)

    # if loading existing embedding
    elif (cfg.RESUME > -1 and cfg.STAGE == 2):
        emb_file = filename_parser.get_embedding_file(
            cfg.EXPERIMENT_PATH, cfg.EMBEDDINGS.EMBEDDING_ID)

        logger.info(f"loading existing node embeddings from: '{emb_file}'")
        embedding_ckpt = torch.load(emb_file)
        embedding = embedding_ckpt['embedding']

    # load embedding into graph
    graph.ndata[cfg.GRAPH_KEYS.EMB_KEY] = embedding.to(graph.device)
    return graph


def perform_tasks(cfg, graph):
    result_dict = {}
    if cfg.TASK in ['linkpred', 'all']:
        logger.info('performing link prediction task..')
        results = perform_link_prediction_single(
            graph, cfg.GRAPH_KEYS.GROUP_KEY, test_size=cfg.LINK_PREDICTION.TEST_SIZE, seed=cfg.SEED)
        result_dict['linkpred'] = results

    if cfg.TASK in ['infmax', 'all']:
        logger.info('performing influence maximization task..')
        results = perform_influence_maximization_single(
            graph,
            cfg.INFLUENCE_MAXIMIZATION.K,
            cfg.INFLUENCE_MAXIMIZATION.P_INFECTION,
            cfg.GRAPH_KEYS.GROUP_KEY,
            cfg.GRAPH_KEYS.EMB_KEY
        )
        result_dict['infmax'] = results

    if cfg.TASK in ['nodeclass', 'all']:
        logger.info('performing node classification task..')
        results = perform_node_classification_single(graph, cfg.GRAPH_KEYS.GROUP_KEY, "college", cfg.GRAPH_KEYS.EMB_KEY,
                                                     cfg.NODE_CLASSIFICATION.TEST_SIZE, cfg.NODE_CLASSIFICATION.N_NEIGHBORS)
        result_dict['nodeclass'] = results

    return result_dict


def perform_experiment(cfg):
    """Run full pipeline for 1 experiment config. 

    Using the RESUME and STAGE arguments this pipeline can be further resumed, 
    see default.py for more information.

    The pipeline consists of the following steps 
    STAGE 0 (fresh run)
    1. Load or synthesize graph.
    STAGE 1
    2. Reweight graph.
    3. Train node embeddings.
    STAGE 2
    4. Perform downstream task(s).
    5. Compute metrics.

    Args:
        experiment_config (yacs.config.CfgNode): config containing all experiment settings

    """
    logger
    graph = load_or_construct_graph(cfg)

    logger.info(f"graph shape, num_nodes={graph.num_nodes()}, num_edges={graph.num_edges()}")
    prior_weight_median = graph.edata[cfg.GRAPH_KEYS.PRIOR_WEIGHTS_KEY].median()
    graph = reweight_edges(cfg, graph)
    weight_median = graph.edata[cfg.GRAPH_KEYS.WEIGHT_KEY].median()
    logger.info(f"absolute difference between median of weights after reweighting: {torch.absolute(prior_weight_median - weight_median)}")
    graph = add_embeddings(cfg, graph)
    result_dict = perform_tasks(cfg, graph)
    return result_dict


def parse_infmax_runs(results):
    if 'infmax' in results[0]:
        logger.info("=" * 80)
        logger.info("results for task influence maximization")
        task_res = [r['infmax'] for r in results]
        infected_nodes = []
        infected_ratios_by_group = []
        disparities = []

        for run_idx in range(len(task_res)):
            logger.info("-" * 80)
            logger.info(f"run {run_idx}:")
            logger.info(
                f"    infected_nodes_fraction: {task_res[run_idx]['infected_nodes_fraction']}")
            # print(f"    infected_ratios_by_group: {task_res[run_idx]['infected_ratios_by_group']}")
            logger.info(f"    disparity: {task_res[run_idx]['disparity']}")

            infected_nodes.append(task_res[run_idx]['infected_nodes_fraction'])
            # infected_ratios_by_group.append(task_res[run_idx]['infected_ratios_by_group'])
            disparities.append(task_res[run_idx]['disparity'])

        logger.info("=" * 80)
        logger.info("final mean of all runs:")
        logger.info(f"    infected_nodes_fraction: {np.mean(infected_nodes)}")
        # print(f"infected_ratios_by_group: {np.mean(infected_ratios_by_group)}")
        logger.info(f"    disparities: {np.mean(disparities)}")
        logger.info("")
        logger.info("final std/var of all runs:")
        logger.info(
            f"    infected_nodes_fraction: std={np.std(infected_nodes)}, var={np.var(infected_nodes)}")
        logger.info(f"    disparities: std={np.std(disparities)}, var={np.var(disparities)}")

        logger.info("-" * 80)
        logger.info("infected_nodes_fraction values raw:")
        logger.info(f"{infected_nodes}")
        logger.info("disparity values raw:")
        logger.info(f"{disparities}")

        logger.info("=" * 80)


def parse_linkpred_runs(results):
    if 'linkpred' in results[0]:
        logger.info("=" * 80)
        logger.info("results for task link prediciton")
        task_res = [r['linkpred'] for r in results]
        accuracies = []
        disparities = []

        for run_idx in range(len(task_res)):
            logger.info("-" * 80)
            logger.info(f"run {run_idx}:")
            logger.info(f"    accuracy: {task_res[run_idx]['accuracy']}")
            logger.info(f"    disparity: {task_res[run_idx]['disparity']}")

            accuracies.append(task_res[run_idx]['accuracy'])
            disparities.append(task_res[run_idx]['disparity'])

        logger.info("=" * 80)
        logger.info("final average of all runs:")
        logger.info(f"    accuracy: {np.mean(accuracies)}")
        logger.info(f"    disparity: {np.mean(disparities)}")
        logger.info("")
        logger.info("final std/var of all runs:")
        logger.info(f"    accuracy: std={np.std(accuracies)}, var={np.var(accuracies)}")
        logger.info(f"    disparity: std={np.std(disparities)}, var={np.var(disparities)}")

        logger.info("-" * 80)
        logger.info("accuracy values raw:")
        logger.info(f"{accuracies}")
        logger.info("disparity values raw:")
        logger.info(f"{disparities}")

        logger.info("=" * 80)


def parse_nodeclass_runs(results):
    if 'nodeclass' in results[0]:
        logger.info("=" * 80)
        logger.info("results for task node classification")
        task_res = [r['nodeclass'] for r in results]
        accuracies = []
        disparities = []

        for run_idx in range(len(task_res)):
            logger.info("-" * 80)
            logger.info(f"run {run_idx}:")
            logger.info(f"    accuracy: {task_res[run_idx]['accuracy_mean']}")
            logger.info(f"    disparity: {task_res[run_idx]['disparity']}")

            accuracies.append(task_res[run_idx]['accuracy_mean'])
            disparities.append(task_res[run_idx]['disparity'])

        logger.info("=" * 80)
        logger.info("final average of all runs:")
        logger.info(f"    accuracy: {np.mean(accuracies)}")
        logger.info(f"    disparities: {np.mean(disparities)}")

        logger.info("")
        logger.info("final std/var of all runs:")
        logger.info(f"    accuracy: std={np.std(accuracies)}, var={np.var(accuracies)}")
        logger.info(f"    disparity: std={np.std(disparities)}, var={np.var(disparities)}")

        logger.info("-" * 80)
        logger.info("accuracy values raw:")
        logger.info(f"{accuracies}")
        logger.info("disparity values raw:")
        logger.info(f"{disparities}")

        logger.info("=" * 80)


def perform_multi_experiment(cfg):
    results = []
    for run_idx in range(cfg.RUNS):
        logger.info("=" * 80)
        logger.info(f"                        Performing experiment trial {run_idx+1}/{cfg.RUNS}")
        logger.info("=" * 80)
        result_dict = perform_experiment(cfg)
        results.append(result_dict)

    # parse results for multiple runs
    parse_infmax_runs(results)
    parse_linkpred_runs(results)
    parse_nodeclass_runs(results)
    return results


def write_run_all_results(results, target_path):
    header = ['task', 'dataset', 'embedding_method',
              'weighting_method', 'metric', 'value', 'n_runs', 'extra']
    row_list = []

    for exp_idx, [exp_result, cfg] in enumerate(results):
        task = cfg.TASK
        dataset = cfg.DATASET
        emb_method = cfg.EMBEDDINGS.METHOD
        graph_w_method = cfg.GRAPH_WEIGHTING.METHOD

        if dataset == 'synthetic':
            dataset += str(len(cfg.SYNTHETIC.NODES))

        if task == 'nodeclass':
            exp_result = [r['nodeclass'] for r in exp_result]
            disp = [r['disparity'] for r in exp_result]
            acc = [r['accuracy_mean'] for r in exp_result]
            result_dict = {
                "mean_accuracy": np.mean(acc),
                "mean_disparity": np.mean(disp),
                "std_accuracy": np.std(acc),
                "std_disparity": np.std(disp),
                "var_accuracy": np.var(acc),
                "var_disparity": np.var(disp),
            }
        if task == 'linkpred':
            exp_result = [r['linkpred'] for r in exp_result]
            disp = [r['disparity'] for r in exp_result]
            acc = [r['accuracy'] for r in exp_result]
            result_dict = {
                "mean_accuracy": np.mean(acc),
                "mean_disparity": np.mean(disp),
                "std_accuracy": np.std(acc),
                "std_disparity": np.std(disp),
                "var_accuracy": np.var(acc),
                "var_disparity": np.var(disp)
            }
        if task == 'infmax':
            exp_result = [r['infmax'] for r in exp_result]
            inf_frac = [r['infected_nodes_fraction'] for r in exp_result]
            disp = [r['disparity'] for r in exp_result]
            result_dict = {
                "mean_infected_nodes_fraction": np.mean(inf_frac),
                "mean_disparity": np.mean(disp),
                "std_infected_nodes_fraction": np.std(inf_frac),
                "std_disparity": np.std(disp),
                "var_infected_nodes_fraction": np.var(inf_frac),
                "var_disparity": np.var(disp)
            }
        extra = cfg.EXTRA_LOG_NAME
        for metric, value in result_dict.items():
            row_list.append([task, dataset, emb_method, graph_w_method,
                            metric, value, len(exp_result), extra])

    df = pd.DataFrame.from_records(row_list, columns=header)
    df.to_csv(target_path, sep=';')


def reproduce(cfg):
    """Reproduce all the experiments from the repository. 

    This function reads all configs from the experiment folder that 
    encompass all the results from the original crosswalk paper 
    in one function call. For individual experiments it is recommended
    to use the .yml config files instead to run one experiment at a time.
    """
    all_experiments = glob.glob(cfg.RUN_ALL + "/*.yml")
    all_experiments = sorted(all_experiments)

    print("#" * 80)
    print("reproducing all experiments!")
    print(f"     found {len(all_experiments)} experiment configs in folder.")
    print("#" * 80)
    run_all_results = []
    for exp_idx, experiment_fp in enumerate(all_experiments):
        print(f"running experiment {exp_idx+1}/{len(all_experiments)}")
        cfg = get_cfg_defaults()
        fake_args = type("FakeArgs", (dict, object), {"cfg": experiment_fp, "opts": []})
        cfg = update_config(cfg, fake_args, freeze=False)

        setup_folder(cfg)
        setup_logger(cfg)
        try:
            if cfg.RUNS == 1:
                results = perform_experiment(cfg)
                parse_infmax_runs([results])
                parse_linkpred_runs([results])
                parse_nodeclass_runs([results])
                run_all_results.append([results, cfg])
            else:
                logger.info(
                    f"Repeating experiment config: \n'{experiment_fp}' \n{cfg.RUNS} times and averaging results")
                results = perform_multi_experiment(cfg)
                run_all_results.append([results, cfg])
        except Exception as e:
            logger.info(
                f"something went wrong when parsing '{experiment_fp}':\n{e} continuing to next experiment")

        now = datetime.datetime.now()
        logger.info("+" * 80)
        logger.info(f"             Experiment end time: {now}")
        logger.info("+" * 80)
        # remove handlers
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()

    now = datetime.datetime.now().strftime(format="%Y-%m-%d_%H-%M")
    write_run_all_results(run_all_results, str(
        Path(__file__).parents[1] / f'run_all_results_{now}.csv'))


def setup_folder(cfg):
    """Setup experiment folder based on config.

    Args:
        cfg (cfg): config with all options
    """
    # generate experiment folder to save results in.
    experiment_folder = filename_parser.generate_filename(cfg)
    experiment_path = Path(__file__).parents[1] / cfg.BASEDIR / experiment_folder

    cfg.EXPERIMENT_PATH = str(experiment_path)
    os.makedirs(cfg.EXPERIMENT_PATH)

    # dump config to experiment_path
    with open(Path(cfg.EXPERIMENT_PATH) / "config.yml", "w") as f:
        str_cfg = cfg.dump()
        f.write(str_cfg)

    # setup folder structure within experiment folder
    os.makedirs(Path(cfg.EXPERIMENT_PATH) / "graph")
    os.makedirs(Path(cfg.EXPERIMENT_PATH) / "embeddings")
    os.makedirs(Path(cfg.EXPERIMENT_PATH) / "visualization")


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="",
                        type=str)
    parser.add_argument('--run-all',
                        help='run all the experiment configs in the given directory',
                        type=str)
    parser.add_argument('--opts',
                        help="Modify config options using the command-line",
                        default=[],
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    if args.cfg:
        cfg = update_config(cfg, args, freeze=False)

    cfg.RUN_ALL = args.run_all
    print(cfg.RUN_ALL)
    cfg.CFG_PATH = args.cfg

    # only setup everything here if a single experiment is run otherwise it is done in the reproduce function.
    if not cfg.RUN_ALL:
        if cfg.RESUME > -1:
            cfg.EXPERIMENT_PATH = filename_parser.get_experiment_folder(cfg.BASEDIR, cfg.RESUME)
        else:
            setup_folder(cfg)
        setup_logger(cfg)

    cfg.freeze()
    return cfg


def setup_logger(cfg):
    """Setup file and console stream handlers to log to the correct experiment folders.

    This file method is called once when the script is ran for 1 config.yml and multiple times
    if parsing the whole folder.

    Args:
        cfg (cfg): config used to determine logfile location.
    """
    global logger
    logfile = Path(cfg.EXPERIMENT_PATH) / 'log.txt'
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    fh = logging.FileHandler(logfile, "w+")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    logger.info("=" * 80)
    logger.info("-" * 80)
    logger.info(f"|{' ' * 25}CROSSWALK REPRODUCTION SUITE{' ' * 25}|")
    logger.info("-" * 80)
    logger.info("=" * 80)

    now = datetime.datetime.now()
    logger.info("+" * 80)
    logger.info(f"             Experiment start time: {now}")
    logger.info("+" * 80)


def main():
    cfg = parse_args()
    set_seed(cfg.SEED)

    if cfg.RUN_ALL:
        reproduce(cfg)
    else:

        if cfg.RESUME > -1:
            stage_name = {
                1: "reweighting graph and training embeddings",
                2: "performing downstream tasks"
            }
            logger.info(
                F"Continuing experiment_{cfg.RESUME} from stage {cfg.STAGE}, {stage_name[cfg.STAGE]}")

        if cfg.RUNS == 1:
            results = perform_experiment(cfg)
            parse_infmax_runs([results])
            parse_linkpred_runs([results])
            parse_nodeclass_runs([results])
        else:
            logger.info(
                f"Repeating experiment config: \n       '{cfg.CFG_PATH}' \n{cfg.RUNS} times and averaging results")
            results = perform_multi_experiment(cfg)

        now = datetime.datetime.now()
        logger.info("+" * 80)
        logger.info(f"             Experiment end time: {now}")
        logger.info("+" * 80)


if __name__ == '__main__':
    main()
