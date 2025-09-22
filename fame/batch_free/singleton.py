import keras
import numpy as np
from fame.abstract_domain.abstract import (
    get_abstract_output_domain_singleton, get_abstract_output_domain
)
from typing import Union


def free_domain_with_abstract_interpretation_singleton(
    model: keras.models.Model,
    input_sample: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    xai_indices: list[int],
    free_indices: list[int],
    potential_candidates:Union[list[int],None]=None, # subset of remaining indices where we know there is potential to be freed
    channel: int = 1,
    data_format: str = "channels_first",
    n_class: int = 10,
    decomon_model: keras.models.Model = None,
) -> tuple[list[int], list[int]]:
    # expand one dimension in the set of remaining features to the
    n_in_with_channel: int = input_sample.shape[-1]
    n_in_wo_channel: int = int(input_sample.shape[-1]/channel)

    remaining_indices: list[int]
    if potential_candidates is None:
        remaining_indices = [i for i in range(n_in_wo_channel) if i not in xai_indices + free_indices]
    else:
        remaining_indices = potential_candidates

    
    upper: np.ndarray
    upper = get_abstract_output_domain_singleton(
        model=model,
        input_sample=input_sample,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        xai_indices=xai_indices,
        free_indices=free_indices,
        remaining_indices=remaining_indices,
        channel=channel,
        data_format=data_format,
        n_class=n_class,
        decomon_model=decomon_model,
    )


    if np.min(np.max(upper, -1)) <= 0:
        top_index = np.argmin(np.max(upper, -1))
        other_indices = [i for i in range(len(upper)) if np.max(upper, -1)[i]<=0 and i !=top_index]
        return [remaining_indices[top_index]], [remaining_indices[j] for j in other_indices]
    else:
        return [], []



##### binary search
def free_with_binary_search(model: keras.models.Model,
    input_sample: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    xai_indices: list[int],
    free_indices: list[int],
    potential_candidates:Union[list[int],None]=None, # subset of remaining indices where we know there is potential to be freed
    channel: int = 1,
    data_format: str = "channels_first",
    n_class: int = 10,
    decomon_model: keras.models.Model = None)->list[int]:

    # step 1: identify singleton that could be free
    best_singleton:list[int]
    other_singleton:list[int]
    best_singleton, other_singleton = free_domain_with_abstract_interpretation_singleton(
        model=model,
        input_sample=input_sample,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        xai_indices=xai_indices,
        free_indices=free_indices,
        potential_candidates=potential_candidates,
        channel=channel,
        data_format=data_format,
        n_class=n_class,
        decomon_model=decomon_model
    )

    if len(other_singleton):
        # create an order
        traversal_order_indices:list[int] = best_singleton+other_singleton
        # try to free everything at once
        upper :np.array = get_abstract_output_domain(model=model,
                                                    input_sample=input_sample,
                                                    lower_bound=lower_bound,
                                                    upper_bound=upper_bound,
                                                    xai_indices=xai_indices,
                                                    free_indices=free_indices+traversal_order_indices,
                                                    channel=channel,
                                                    data_format=data_format,
                                                    n_class=n_class,
                                                    decomon_model=decomon_model
                                                    ) # (1, n_out)
        is_safe:bool = np.max(upper)<=0
        if is_safe:
            return traversal_order_indices
        else:
            # split in half
            n_singleton:int = len(traversal_order_indices)
            n_singleton_half:int = int(n_singleton/2)
            traversal_order_indices_part_0:list[int] = traversal_order_indices[:n_singleton_half]
            traversal_order_indices_part_1:list[int] = traversal_order_indices[n_singleton_half:]
            # free as many as possible from this list with recursive calls of free_with_binary_search
            singleton_indices_part_0 = free_with_binary_search(model=model,
                                                        input_sample=input_sample,
                                                        lower_bound=lower_bound,
                                                        upper_bound=upper_bound,
                                                        xai_indices=xai_indices,
                                                        free_indices=free_indices,
                                                        potential_candidates=traversal_order_indices_part_0,
                                                        channel=channel,
                                                        data_format=data_format,
                                                        n_class=n_class,
                                                        decomon_model=decomon_model
            )
            # considering this singleton indices as part of the free indices try to free as much as possible the rest
            singleton_indices_part_1 = free_with_binary_search(model=model,
                                                        input_sample=input_sample,
                                                        lower_bound=lower_bound,
                                                        upper_bound=upper_bound,
                                                        xai_indices=xai_indices,
                                                        free_indices=free_indices+singleton_indices_part_0,
                                                        potential_candidates=traversal_order_indices_part_1,
                                                        channel=channel,
                                                        data_format=data_format,
                                                        n_class=n_class,
                                                        decomon_model=decomon_model
            )
            return singleton_indices_part_0+singleton_indices_part_1
    else:
        return best_singleton
    
def free_with_singleton_search(model: keras.models.Model,
    input_sample: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    xai_indices: list[int],
    free_indices: list[int],
    potential_candidates:Union[list[int],None]=None, # subset of remaining indices where we know there is potential to be freed
    channel: int = 1,
    data_format: str = "channels_first",
    n_class: int = 10,
    decomon_model: keras.models.Model = None)->list[int]:

    # step 1: identify singleton that could be free
    best_singleton:list[int]
    other_singleton:list[int]
    best_singleton, other_singleton = free_domain_with_abstract_interpretation_singleton(
        model=model,
        input_sample=input_sample,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        xai_indices=xai_indices,
        free_indices=free_indices,
        potential_candidates=potential_candidates,
        channel=channel,
        data_format=data_format,
        n_class=n_class,
        decomon_model=decomon_model
    )

    singleton_solutions:list[int]=best_singleton

    while len(other_singleton):
        best_singleton, other_singleton = free_domain_with_abstract_interpretation_singleton(
        model=model,
        input_sample=input_sample,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        xai_indices=xai_indices,
        free_indices=free_indices,
        potential_candidates=other_singleton,
        channel=channel,
        data_format=data_format,
        n_class=n_class,
        decomon_model=decomon_model
        )  
        singleton_solutions+=best_singleton 
    
    return singleton_solutions
