import time
import os
import keras
import numpy as np
import pandas as pd
from fame import free_at_once_k_features, free_iteratively_k_features, find_closest_xai
import gc

## Experiment A: MILP versus Greedy


### A.1 in one round: what is the running time and largest free set obtained when using milp or greedy
def exp_A_1(
    model: keras.models.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    indices: list[int],
    eps: float,
    dataframe_repository: str,
    dataframe_filename: str,
    channel: int = 1,
    data_format: str = "channels_first",
    n_class: int = 10,
    verbose: int = 0,
    sleep_time:int = 1, # one second between each run
):
    start_time: float
    end_time: float

    # create dico structure for the pandas dataframe
    dico = dict()
    dico["index"] = []
    dico["label"] = []
    dico["milp_size"] = []
    dico["milp_time"] = []
    dico["greedy_size"] = []
    dico["greedy_time"] = []
    dico["greedy_2_milp"] = []

    n_in_wo_channel = int(x_test.shape[-1] / channel)
    xai_indices = []
    free_indices = []
    cardinality = np.array([i for i in range(1, n_in_wo_channel)])

    for index in indices:
        if verbose:
            print("ongoing index", index)

        array_greedy_2_milp = np.zeros_like(cardinality)
        abstract_domain_time = []  # average over milp and greedy

        for coeff, method in zip([1, -1], ["milp", "greedy"]):
            
            time.sleep(sleep_time)
            gc.collect()

            # define input sample and local robustness region
            input_sample = x_test[index]
            gt_label = y_test[index]
            lower_bound_input = np.maximum(input_sample - eps, 0 * input_sample)
            upper_bound_input = np.minimum(input_sample + eps, 0 * input_sample + 1)
            start_time = time.time()
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
                method=method,
                verbose=int(verbose > 1),
            )
            end_time = time.time()
            # update array_greedy_2_milp to compute the worst case predicted distance between milp and greedy
            array_greedy_2_milp = array_greedy_2_milp + coeff * abstract_set.sum(-1)
            # for each method compute the largest abstract free set
            xai_size = np.max(abstract_set.sum(-1))
            running_time = end_time - start_time
            if verbose:
                print("{} time".format(method), running_time)

            dico["{}_size".format(method)].append(xai_size)
            dico["{}_time".format(method)].append(running_time)
        # add abstract domain average running time and distance between milp and greedy
        dico["index"].append(index)
        dico["label"].append(gt_label)
        dico["greedy_2_milp"].append(np.max(array_greedy_2_milp))

        # record at every sample
        # Create the DataFrame
        df = pd.DataFrame(dico)
        df.to_csv("{}/{}.csv".format(dataframe_repository, dataframe_filename), index=False)
    return dico


### A.2 iteratively, what is the running time and largest free set obtained when using milp or greedy
def exp_A_2(
    model: keras.models.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    indices: list[int],
    eps: float,
    dataframe_repository: str,
    dataframe_filename: str,
    channel: int = 1,
    data_format: str = "channels_first",
    n_class: int = 10,
    verbose: int = 0,
    sleep_time:int = 1, # one second between each run
):
    start_time: float
    end_time: float
    array_greedy_2_milp: np.ndarray
    abstract_set: list[int]

    # create dico structure for the pandas dataframe
    dico = dict()
    dico["index"] = []
    dico["label"] = []
    dico["milp_size"] = []
    dico["milp_time"] = []
    dico["greedy_size"] = []
    dico["greedy_time"] = []
    dico["greedy_2_milp"] = []

    n_in_wo_channel = int(x_test.shape[-1] / channel)

    for index in indices:
        if verbose:
            print("ongoing index", index)
        cardinality = np.array([i for i in range(1, n_in_wo_channel)])

        array_greedy_2_milp = np.zeros_like(cardinality)

        for coeff, method in zip([1, -1], ["milp", "greedy"]):
            time.sleep(sleep_time)
            gc.collect()

            # define input sample and local robustness region
            input_sample = np.copy(x_test[index] + 0.0)
            gt_label = np.copy(y_test[index] + 0)
            xai_indices = []
            free_indices = []
            start_time = time.time()

            abstract_set = free_iteratively_k_features(
                model=model,
                gt_label=gt_label,
                input_sample=input_sample,
                eps=eps,
                xai_indices=xai_indices,
                free_indices=free_indices,
                channel=channel,
                data_format=data_format,
                n_class=n_class,
                method=method,
                verbose=int(verbose > 1),
            )
            end_time = time.time()
            # update array_greedy_2_milp to compute the worst case predicted distance between milp and greedy
            array_greedy_2_milp = array_greedy_2_milp + coeff * len(abstract_set)
            # for each method compute the largest abstract free set
            xai_size = len(abstract_set)
            running_time = end_time - start_time
            if verbose:
                print("{} time".format(method), running_time)

            dico["{}_size".format(method)].append(xai_size)
            dico["{}_time".format(method)].append(running_time)
        # add abstract domain average running time and distance between milp and greedy
        dico["index"].append(index)
        dico["label"].append(gt_label)
        dico["greedy_2_milp"].append(np.max(array_greedy_2_milp))

        # record at every sample
        # Create the DataFrame
        df = pd.DataFrame(dico)
        df.to_csv("{}/{}.csv".format(dataframe_repository, dataframe_filename), index=False)
    return dico

def exp_A_2_no_overwrite(
    model: keras.models.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    indices: list[int],
    eps: float,
    dataframe_repository: str,
    dataframe_filename: str,
    channel: int = 1,
    data_format: str = "channels_first",
    n_class: int = 10,
    verbose: int = 0,
    sleep_time:int = 1, # one second between each run
):
    start_time: float
    end_time: float
    array_greedy_2_milp: np.ndarray
    abstract_set: list[int]

    # create dico structure for the pandas dataframe
    dico = dict()
    dico["index"] = []
    dico["label"] = []
    dico["milp_size"] = []
    dico["milp_time"] = []
    dico["greedy_size"] = []
    dico["greedy_time"] = []
    dico["greedy_2_milp"] = []

    n_in_wo_channel = int(x_test.shape[-1] / channel)

    for index in indices:
        if verbose:
            print("ongoing index", index)
        cardinality = np.array([i for i in range(1, n_in_wo_channel)])

        array_greedy_2_milp = np.zeros_like(cardinality)

        for coeff, method in zip([1, -1], ["milp", "greedy"]):
            time.sleep(sleep_time)
            gc.collect()

            # define input sample and local robustness region
            input_sample = np.copy(x_test[index] + 0.0)
            gt_label = np.copy(y_test[index] + 0)
            xai_indices = []
            free_indices = []
            start_time = time.time()

            abstract_set = free_iteratively_k_features(
                model=model,
                gt_label=gt_label,
                input_sample=input_sample,
                eps=eps,
                xai_indices=xai_indices,
                free_indices=free_indices,
                channel=channel,
                data_format=data_format,
                n_class=n_class,
                method=method,
                verbose=int(verbose > 1),
            )
            end_time = time.time()
            # update array_greedy_2_milp to compute the worst case predicted distance between milp and greedy
            array_greedy_2_milp = array_greedy_2_milp + coeff * len(abstract_set)
            # for each method compute the largest abstract free set
            xai_size = len(abstract_set)
            running_time = end_time - start_time
            if verbose:
                print("{} time".format(method), running_time)

            dico["{}_size".format(method)].append(xai_size)
            dico["{}_time".format(method)].append(running_time)
        # add abstract domain average running time and distance between milp and greedy
        dico["index"].append(index)
        dico["label"].append(gt_label)
        dico["greedy_2_milp"].append(np.max(array_greedy_2_milp))

        # record at every sample
        # Create the DataFrame
        if os.path.isfile("{}/{}.csv".format(dataframe_repository, dataframe_filename)):
            df_row = pd.DataFrame(dico)
            df_before = pd.read_csv("{}/{}.csv".format(dataframe_repository, dataframe_filename))
            df = pd.concat([df_before, df_row], ignore_index=True)
        else:
            df = pd.DataFrame(dico)
        df.to_csv("{}/{}.csv".format(dataframe_repository, dataframe_filename), index=False)


def exp_B_no_overwrite(
    model: keras.models.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    indices: list[int],
    eps: float,
    dataframe_repository: str,
    dataframe_filename: str,
    channel: int = 1,
    data_format: str = "channels_first",
    method="greedy",
    attack="fgsm",
    traversal_order="greedy",
    device="mps",
    n_class: int = 10,
    verbose: int = 0,
):
    start_time: float
    end_time: float
    xai_indices:list[int]
    free_indices:list[int]
    remaining_indices:list[int]

    # create dico structure for the pandas dataframe
    dico = dict()
    dico["index"] = []
    dico["label"] = []
    dico["greedy_size_min"] = []
    dico["greedy_size_max"] = []
    dico["greedy_time"] = []
    dico["xai_indices"] = []
    dico["free_indices"] = []
    dico["remaining_indices"]=[]
    dico["dist_2_minimal_xai"]=[]

    for index in indices:
        if verbose:
            print("ongoing index", index)

        # define input sample and local robustness region
        input_sample = np.copy(x_test[index] + 0.0)
        gt_label = np.copy(y_test[index] + 0)
        xai_indices = []
        free_indices = []
        start_time = time.time()

        free_indices = free_iteratively_k_features(
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
        )
        # compute distance to minimal explanation
        potential_xai, remaining_indices = find_closest_xai(model=model, 
                    gt_label=gt_label, 
                    input_sample=input_sample, 
                    eps=eps, 
                    xai_indices=xai_indices, 
                    free_indices=free_indices, 
                    method=attack,
                    device=device,
                    channel=channel,
                    data_format=data_format,
                    n_class=n_class,
                    traversal_order=traversal_order
                    )
        end_time = time.time()
        running_time = end_time - start_time
        if verbose:
            print("{} time".format(method), running_time)

        dico["{}_size_min".format(method)].append(len(potential_xai))
        dico["{}_size_max".format(method)].append(len(potential_xai)+len(remaining_indices))
        dico["free_indices"].append(free_indices)
        dico["xai_indices"].append(potential_xai)
        dico["remaining_indices"].append(remaining_indices)
        dico["{}_time".format(method)].append(running_time)
        dico["dist_2_minimal_xai"].append(len(remaining_indices))
        # add abstract domain average running time and distance between milp and greedy
        dico["index"].append(index)
        dico["label"].append(gt_label)

        # record at every sample
        # Create the DataFrame
        if os.path.isfile("{}/{}.csv".format(dataframe_repository, dataframe_filename)):
            df_row = pd.DataFrame(dico)
            df_before = pd.read_csv("{}/{}.csv".format(dataframe_repository, dataframe_filename))
            df = pd.concat([df_before, df_row], ignore_index=True)
        else:
            #import pdb; pdb.set_trace()
            df = pd.DataFrame(dico)
        df.to_csv("{}/{}.csv".format(dataframe_repository, dataframe_filename), index=False)

