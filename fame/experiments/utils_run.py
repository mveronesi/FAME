import gc
import os
import time
from typing import Literal
from pathlib import Path
import keras
import numpy as np
import pandas as pd
from fame import (
    free_at_once_k_features,
    free_iteratively_k_features,
    free_at_once_k_features_l2,
    free_iteratively_k_features_l2,
)

def save_results_to_csv(dico: dict, output_path: Path):
    """
    Save the results to a CSV file.
    If the file already exists, it will be overwritten.
    """
    if output_path.exists():
        mode = 'a'
        header = False
    else:
        mode = 'w'
        header = True
    df = pd.DataFrame(dico)
    df.to_csv(output_path, index=False, mode=mode, header=header)


### A.1 in one round: running time and largest free set using greedy only
def exp_A_1(
    model: keras.models.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    indices: list[int],
    eps: float,
    output_path: Path,
    channel: int = 1,
    data_format: str = "channels_first",
    n_class: int = 10,
    verbose: int = 0,
    sleep_time: int = 1,  # one second between each run
    norm: Literal["linf", "l2"] = "linf",
    means=None, 
    stddev=None
):
    start_time: float
    end_time: float

    # create dico structure for the pandas dataframe
    dico = dict()
    dico["index"] = []
    dico["label"] = []
    dico["freed_features"] = []
    dico["greedy_time"] = []

    n_in_wo_channel = int(x_test.shape[-1] / channel)
    xai_indices = []
    free_indices = []
    cardinality = np.array([i for i in range(1, n_in_wo_channel)])

    for index in indices:
        if verbose:
            print("ongoing index", index)

        time.sleep(sleep_time)
        gc.collect()

        # define input sample and local robustness region
        input_sample = x_test[index]
        gt_label = y_test[index]

        if means is None or stddev is None:
            lower_bound_input = np.maximum(input_sample - eps, 0 * input_sample)
            upper_bound_input = np.minimum(input_sample + eps, 0 * input_sample + 1)
        else:
            lower_bound_input = np.maximum(input_sample - eps, - (means / stddev))
            upper_bound_input = np.minimum(input_sample + eps, ((1 - means) / stddev))

        start_time = time.time()
        if norm == "linf":
            abstract_set = free_at_once_k_features(
                model=model,
                gt_label=gt_label,
                input_sample=input_sample,
                lower_bound_input=lower_bound_input,
                upper_bound_input=upper_bound_input,
                xai_indices=xai_indices,
                free_indices=free_indices,
                cardinality=cardinality,
                channel=channel,
                data_format=data_format,
                n_class=n_class,
                method="greedy",
                verbose=int(verbose > 1),
            )
        elif norm == "l2":
            abstract_set = free_at_once_k_features_l2(
                model=model,
                gt_label=gt_label,
                input_sample=input_sample,
                lower_bound_input=lower_bound_input,
                upper_bound_input=upper_bound_input,
                eps=eps,
                xai_indices=xai_indices,
                free_indices=free_indices,
                cardinality=cardinality,
                channel=channel,
                data_format=data_format,
                n_class=n_class,
                verbose=int(verbose > 1),
            )
        else:
            raise ValueError(f"unknown norm {norm}")
        
        end_time = time.time()
        xai_size = np.max(abstract_set.sum(-1))
        running_time = end_time - start_time

        if verbose:
            print("greedy time", running_time)

        dico["freed_features"].append(xai_size)
        dico["greedy_time"].append(running_time)

        dico["index"].append(index)
        dico["label"].append(gt_label)

        # record at every sample
        # Create the DataFrame
        save_results_to_csv(dico, output_path)
    return dico


### A.2 iteratively: running time and largest free set using greedy only
def exp_A_2(
    model: keras.models.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    indices: list[int],
    eps: float,
    output_path: Path,
    channel: int = 1,
    data_format: str = "channels_first",
    n_class: int = 10,
    verbose: int = 0,
    sleep_time: int = 1,  # one second between each run
    norm: Literal["linf", "l2"] = "linf",
    means = None, 
    stddev = None
):
    start_time: float
    end_time: float
    abstract_set: list[int]

    # create dico structure for the pandas dataframe
    dico = dict()
    dico["index"] = []
    dico["label"] = []
    dico["freed_features"] = []
    dico["greedy_time"] = []
    n_in_wo_channel = int(x_test.shape[-1] / channel)
    
    for index in indices:
        if verbose:
            print("ongoing index", index)

        time.sleep(sleep_time)
        gc.collect()

        # define input sample and local robustness region
        input_sample = np.copy(x_test[index] + 0.0)
        gt_label = np.copy(y_test[index] + 0)
        xai_indices = []
        free_indices = []
        start_time = time.time()

        if norm == "linf":
            abstract_set, singleton_set = free_iteratively_k_features(
                model=model,
                gt_label=gt_label,
                input_sample=input_sample,
                eps=eps,
                xai_indices=xai_indices,
                free_indices=free_indices,
                channel=channel,
                data_format=data_format,
                n_class=n_class,
                method="greedy",
                verbose=int(verbose > 1),
                means=means,
                stddev=stddev,
            )
        elif norm == "l2":
            abstract_set, singleton_set = free_iteratively_k_features_l2(
                model=model,
                gt_label=gt_label,
                input_sample=input_sample,
                eps=eps,
                xai_indices=xai_indices,
                free_indices=free_indices,
                channel=channel,
                data_format=data_format,
                n_class=n_class,
                refining_domain=True,
                singleton_refinement=True,
                verbose=int(verbose > 1),
                means=means,
                stddev=stddev,
            )
        else:
            raise ValueError(f"unknown norm {norm}")
        end_time = time.time()
        abstract_set = abstract_set + singleton_set
        xai_size = len(abstract_set)
        running_time = end_time - start_time

        if verbose:
            print("greedy time", running_time)

        dico["freed_features"].append(xai_size)
        dico["greedy_time"].append(running_time)

        dico["index"].append(index)
        dico["label"].append(gt_label)

        # record at every sample
        # Create the DataFrame
        save_results_to_csv(dico, output_path)
    return dico
