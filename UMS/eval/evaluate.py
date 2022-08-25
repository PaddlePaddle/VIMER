"""
# repo originally forked from https://github.com/Confusezius/Deep-Metric-Learning-Baselines
"""
##################################### LIBRARIES ###########################################
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import time
import pickle as pkl
import csv
import matplotlib.pyplot as plt

from scipy.spatial import distance
from sklearn.preprocessing import normalize

from tqdm import tqdm
import eval.auxiliaries as aux
import os
import paddle


def evaluate(dataset, LOG, **kwargs):
    """
    Given a dataset name, applies the correct evaluation function.

    Args:
        dataset: str, name of dataset.
        LOG:     aux.LOGGER instance, main logging class.
        **kwargs: Input Argument Dict, depends on dataset.
    Returns:
        (optional) Computed metrics. Are normally written directly to LOG and printed.
    """
    if dataset in ["Stanford_Online_Products"]:
        ret = evaluate_one_dataset(LOG, **kwargs)
    elif dataset in ["inshop_dataset"]:
        ret = evaluate_query_and_gallery_dataset(LOG, **kwargs)
    else:
        raise Exception("No implementation for dataset {} available!")

    return ret


def evaluate_one_dataset(
    LOG, dataloader, model, args, save=True, give_return=True, epoch=0
):
    """
    Compute evaluation metrics, update LOGGER and print results.

    Args:
        LOG:         aux.LOGGER-instance. Main Logging Functionality.
        dataloader:  PyTorch Dataloader, Testdata to be evaluated.
        model:       PyTorch Network, Network to evaluate.
        args:         argparse.Namespace, contains all training-specific parameters.
        save:        bool, if True, Checkpoints are saved when testing metrics (specifically Recall @ 1) improve.
        give_return: bool, if True, return computed metrics.
        epoch:       int, current epoch, required for logger.
    Returns:
        (optional) Computed metrics. Are normally written directly to LOG and printed.
    """
    start = time.time()
    image_paths = np.array(dataloader.dataset.image_list)
    with paddle.no_grad():
        # Compute Metrics
        (
            F1,
            NMI,
            recall_at_ks,
            feature_matrix_all,
            map_at_r,
        ) = aux.eval_metrics_one_dataset(
            model, dataloader, k_vals=args.k_vals, args=args
        )
        # Make printable summary string.

        result_str = ", ".join(
            "@{0}: {1:.4f}".format(k, rec) for k, rec in zip(args.k_vals, recall_at_ks)
        )
        result_str = (
            "Epoch (Test) {0}: NMI [{1:.4f}] | F1 [{2:.4f}] | Recall [{3}]".format(
                epoch, NMI, F1, result_str
            )
        )

        if LOG is not None:
            LOG.log(
                "val",
                LOG.metrics_to_log["val"],
                [epoch, np.round(time.time() - start), NMI, F1] + recall_at_ks,
            )

    print(result_str)
    if give_return:
        return recall_at_ks, NMI, F1
    else:
        None


def evaluate_query_and_gallery_dataset(
    LOG,
    query_dataloader,
    gallery_dataloader,
    model,
    args,
    save=True,
    give_return=True,
    epoch=0,
):
    """
    Compute evaluation metrics, update LOGGER and print results, specifically for In-Shop Clothes.

    Args:
         LOG:         aux.LOGGER-instance. Main Logging Functionality.
        query_dataloader:    PyTorch Dataloader, Query-testdata to be evaluated.
        gallery_dataloader:  PyTorch Dataloader, Gallery-testdata to be evaluated.
        model:       PyTorch Network, Network to evaluate.
        args:         argparse.Namespace, contains all training-specific parameters.
        save:        bool, if True, Checkpoints are saved when testing metrics (specifically Recall @ 1) improve.
        give_return: bool, if True, return computed metrics.
        epoch:       int, current epoch, required for logger.
     Returns:
        (optional) Computed metrics. Are normally written directly to LOG and printed.
    """
    start = time.time()
    query_image_paths = np.array([x[0] for x in query_dataloader.dataset.image_list])
    gallery_image_paths = np.array(
        [x[0] for x in gallery_dataloader.dataset.image_list]
    )
    with paddle.no_grad():
        # Compute Metri cs.
        (
            F1,
            NMI,
            recall_at_ks,
            query_feature_matrix_all,
            gallery_feature_matrix_all,
            map_at_r,
        ) = aux.eval_metrics_query_and_gallery_dataset(
            model, query_dataloader, gallery_dataloader, k_vals=args.k_vals, args=args
        )

        # Generate printable summary string.
        result_str = ", ".join(
            "@{0}: {1:.4f}".format(k, rec) for k, rec in zip(args.k_vals, recall_at_ks)
        )
        result_str = (
            "Epoch (Test) {0}: NMI [{1:.4f}] | F1 [{2:.4f}] | Recall [{3}]".format(
                epoch, NMI, F1, result_str
            )
        )
        if LOG is not None:
            LOG.log(
                "val",
                LOG.metrics_to_log["val"],
                [epoch, np.round(time.time() - start), NMI, F1] + recall_at_ks,
            )

    print(result_str)
    if give_return:
        return recall_at_ks, NMI, F1
    else:
        None
