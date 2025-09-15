import time

import keras
import numpy as np
import pandas as pd
from fame import free_at_once_k_features, free_iteratively_k_features

## Experiment A: MILP versus Greedy

### A.1 in one round: what is the running time and largest free set obtained when using milp or greedy
def exp_A_1(model:keras.models.Model,
            x_test:np.ndarray,
            y_test:np.ndarray,
            indices:list[int],
            eps:float,
            dataframe_repository:str,
            dataframe_filename:str,
            channel:int=1,
            data_format:str="channels_first",
            n_class:int=10,
            verbose:int=0):

    start_time:float
    end_time:float

    # create dico structure for the pandas dataframe
    dico = dict()
    dico['index']=[]
    dico['label']=[]
    dico['milp_size']=[]
    dico['milp_time']=[]
    dico['greedy_size']=[]
    dico['greedy_time']=[]
    dico['greedy_2_milp']=[]

    n_in_wo_channel = int(x_test.shape[-1]/channel)
    xai_indices=[]
    free_indices=[]
    cardinality = np.array([i for i in range(1,n_in_wo_channel)])

    for index in indices:
        if verbose:
            print('ongoing index', index)

        array_greedy_2_milp = np.zeros_like(cardinality)
        abstract_domain_time = [] # average over milp and greedy

        for coeff, method in zip([1, -1], ['milp', 'greedy']):

            start_time= time.perf_counter()

            # define input sample and local robustness region
            input_sample = x_test[index]
            gt_label = y_test[index]
            lower_bound_input= np.maximum(input_sample - eps, 0*input_sample)
            upper_bound_input= np.minimum(input_sample + eps, 0*input_sample+1)

            abstract_set= free_at_once_k_features(model=model,
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
                                    verbose=verbose,
                                    )
            end_time = time.perf_counter()
            # update array_greedy_2_milp to compute the worst case predicted distance between milp and greedy
            array_greedy_2_milp = array_greedy_2_milp+ coeff*abstract_set.sum(-1)
            # for each method compute the largest abstract free set
            xai_size = np.max(abstract_set.sum(-1))
            running_time = end_time-start_time
            if verbose:
                print('time', running_time)

            dico['{}_size'.format(method)].append(xai_size)
            dico['{}_time'.format(method)].append(running_time)
        # add abstract domain average running time and distance between milp and greedy
        dico['index'].append(index)
        dico['label'].append(gt_label)
        dico['greedy_2_milp'].append(np.max(array_greedy_2_milp))

        # record at every sample
        # Create the DataFrame
        df = pd.DataFrame(dico)
        df.to_csv('{}/{}.csv'.format(dataframe_repository, dataframe_filename), index=False)
    return dico


### A.2 iteratively, what is the running time and largest free set obtained when using milp or greedy
def exp_A_2(model:keras.models.Model,
            x_test:np.ndarray,
            y_test:np.ndarray,
            indices:list[int],
            eps:float,
            dataframe_repository:str,
            dataframe_filename:str,
            channel:int=1,
            data_format:str="channels_first",
            n_class:int=10,
            verbose:int=0):

    start_time:float
    end_time:float
    array_greedy_2_milp:np.ndarray

    # create dico structure for the pandas dataframe
    dico = dict()
    dico['index']=[]
    dico['label']=[]
    dico['milp_size']=[]
    dico['milp_time']=[]
    dico['greedy_size']=[]
    dico['greedy_time']=[]
    dico['greedy_2_milp']=[]

    n_in_wo_channel = int(x_test.shape[-1]/channel)
    xai_indices=[]
    free_indices=[]
    cardinality = np.array([i for i in range(1,n_in_wo_channel)])

    for index in indices:
        if verbose:
            print('ongoing index', index)

        array_greedy_2_milp = np.zeros_like(cardinality)

        for coeff, method in zip([1, -1], ['milp', 'greedy']):

            start_time= time.perf_counter()

            # define input sample and local robustness region
            input_sample = x_test[index]
            gt_label = y_test[index]

            abstract_set= free_iteratively_k_features(model=model,
                                    gt_label=gt_label,
                                    input_sample=input_sample,
                                    eps=eps,
                                    xai_indices=xai_indices,
                                    free_indices=free_indices,
                                    channel=channel,
                                    data_format=data_format,
                                    n_class=n_class,
                                    method=method,
                                    verbose=verbose,
                                    )
            end_time = time.perf_counter()
            # update array_greedy_2_milp to compute the worst case predicted distance between milp and greedy
            array_greedy_2_milp = array_greedy_2_milp+ coeff*abstract_set.sum(-1)
            # for each method compute the largest abstract free set
            xai_size = np.max(abstract_set.sum(-1))
            running_time = end_time-start_time
            if verbose:
                print('time', running_time)

            dico['{}_size'.format(method)].append(xai_size)
            dico['{}_time'.format(method)].append(running_time)
        # add abstract domain average running time and distance between milp and greedy
        dico['index'].append(index)
        dico['label'].append(gt_label)
        dico['greedy_2_milp'].append(np.max(array_greedy_2_milp))

        # record at every sample
        # Create the DataFrame
        df = pd.DataFrame(dico)
        df.to_csv('{}/{}.csv'.format(dataframe_repository, dataframe_filename), index=False)
    return dico

## Experiment B: Distance to abstract minimality
#### store the explanation, remaining indices and free features + the method (fgsm/pgd)
