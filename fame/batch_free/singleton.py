from typing import Union

import keras
import numpy as np
from fame.abstract_domain.abstract import (
    get_abstract_output_domain,
    get_abstract_output_domain_singleton,
)


def free_domain_with_abstract_interpretation_singleton(
    model: keras.models.Model,
    input_sample: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    xai_indices: list[int],
    free_indices: list[int],
    potential_candidates: Union[
        list[int], None
    ] = None,  # subset of remaining indices where we know there is potential to be freed
    channel: int = 1,
    data_format: str = "channels_first",
    n_class: int = 10,
    decomon_model: keras.models.Model = None,
) -> tuple[list[int], list[int]]:
    """Identifies individual features that can be proven robust using abstract interpretation.

    This function tests the robustness of the model when perturbing single features
    ("singletons") from a candidate set, one at a time. For each candidate, it
    constructs a domain where the existing `free_indices` and the single candidate
    feature are perturbed. It then uses `decomon` to formally verify if the
    model's prediction remains constant for every point in that domain.

    Args:
        model: The Keras model to be analyzed.
        input_sample: The nominal input point.
        lower_bound: The lower bounds of the L-infinity perturbation space.
        upper_bound: The upper bounds of the L-infinity perturbation space.
        xai_indices: A list of feature indices to be excluded from the search.
        free_indices: A list of feature indices that are always considered perturbed.
        potential_candidates: An optional subset of features to test. If `None`, all
            features not in `xai_indices` or `free_indices` are tested.
        channel: The number of channels in the input data.
        data_format: The data format, "channels_first" or "channels_last".
        n_class: The number of output classes of the model.
        decomon_model: An optional, pre-compiled decomon model for efficiency.

    Returns:
        A tuple of two lists of feature indices:
        - The first list contains the index of the *most* robust singleton
          feature (the one yielding the most negative certified upper bound).
        - The second list contains the indices of all other singleton features
          that were also found to be robust.
        Returns `([], [])` if no robust singletons are found.
    """
    # expand one dimension in the set of remaining features to the
    n_in_with_channel: int = input_sample.shape[-1]
    n_in_wo_channel: int = int(input_sample.shape[-1] / channel)

    remaining_indices: list[int]
    if potential_candidates is None:
        remaining_indices = [
            i for i in range(n_in_wo_channel) if i not in xai_indices + free_indices
        ]
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
        other_indices = [
            i for i in range(len(upper)) if np.max(upper, -1)[i] <= 0 and i != top_index
        ]
        return [remaining_indices[top_index]], [remaining_indices[j] for j in other_indices]
    else:
        return [], []


##### binary search
def free_with_binary_search(
    model: keras.models.Model,
    input_sample: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    xai_indices: list[int],
    free_indices: list[int],
    potential_candidates: Union[
        list[int], None
    ] = None,  # subset of remaining indices where we know there is potential to be freed
    channel: int = 1,
    data_format: str = "channels_first",
    n_class: int = 10,
    decomon_model: keras.models.Model = None,
) -> list[int]:
    """Finds a maximal robust subset of features using a recursive binary search.

    This function implements a divide-and-conquer algorithm to efficiently find
    a large set of features that can be proven robust when perturbed together.

    The strategy is as follows:
    1.  It first identifies an initial pool of all individually robust features
        (singletons).
    2.  It optimistically checks if this entire pool can be made robust
        simultaneously. If so, it returns the whole set.
    3.  If not, it recursively splits the pool in half. It finds the maximal
        robust subset in the first half.
    4.  It then finds the maximal robust subset in the second half, under the
        assumption that the robust features from the first half are also perturbed.
    5.  The results from the two halves are combined to form the final set.

    Args:
        model: The Keras model to be analyzed.
        input_sample: The nominal input point.
        lower_bound: The lower bounds of the L-infinity perturbation space.
        upper_bound: The upper bounds of the L-infinity perturbation space.
        xai_indices: A list of feature indices to be excluded from the search.
        free_indices: A list of features that are always considered perturbed.
        potential_candidates: An optional subset of features to test.
        channel: The number of channels in the input data.
        data_format: The data format, "channels_first" or "channels_last".
        n_class: The number of output classes.
        decomon_model: An optional, pre-compiled decomon model for efficiency.

    Returns:
        A list containing a large subset of the candidate features that are
        provably robust when perturbed together.
    """

    # step 1: identify singleton that could be free
    best_singleton: list[int]
    other_singleton: list[int]
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
        decomon_model=decomon_model,
    )

    if len(other_singleton):
        # create an order
        traversal_order_indices: list[int] = best_singleton + other_singleton
        # try to free everything at once
        upper: np.array = get_abstract_output_domain(
            model=model,
            input_sample=input_sample,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            #xai_indices=xai_indices,
            free_indices=free_indices + traversal_order_indices,
            channel=channel,
            data_format=data_format,
            n_class=n_class,
            decomon_model=decomon_model,
        )  # (1, n_out)
        is_safe: bool = np.max(upper) <= 0
        if is_safe:
            return traversal_order_indices
        else:
            # split in half
            n_singleton: int = len(traversal_order_indices)
            n_singleton_half: int = int(n_singleton / 2)
            traversal_order_indices_part_0: list[int] = traversal_order_indices[:n_singleton_half]
            traversal_order_indices_part_1: list[int] = traversal_order_indices[n_singleton_half:]
            # free as many as possible from this list with recursive calls of free_with_binary_search
            singleton_indices_part_0 = free_with_binary_search(
                model=model,
                input_sample=input_sample,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                xai_indices=xai_indices,
                free_indices=free_indices,
                potential_candidates=traversal_order_indices_part_0,
                channel=channel,
                data_format=data_format,
                n_class=n_class,
                decomon_model=decomon_model,
            )
            # considering this singleton indices as part of the free indices try to free as much as possible the rest
            singleton_indices_part_1 = free_with_binary_search(
                model=model,
                input_sample=input_sample,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                xai_indices=xai_indices,
                free_indices=free_indices + singleton_indices_part_0,
                potential_candidates=traversal_order_indices_part_1,
                channel=channel,
                data_format=data_format,
                n_class=n_class,
                decomon_model=decomon_model,
            )
            return singleton_indices_part_0 + singleton_indices_part_1
    else:
        return best_singleton


def free_with_singleton_search(
    model: keras.models.Model,
    input_sample: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    xai_indices: list[int],
    free_indices: list[int],
    potential_candidates: Union[
        list[int], None
    ] = None,  # subset of remaining indices where we know there is potential to be freed
    channel: int = 1,
    data_format: str = "channels_first",
    n_class: int = 10,
    decomon_model: keras.models.Model = None,
) -> list[int]:
    """Finds a set of robust features by iteratively and greedily selecting singletons.

    This function implements an iterative greedy algorithm to identify a set of
    features that can be safely perturbed. The process is as follows:
    1.  In each iteration, it finds all individual features that are currently robust
        in the context of the already selected features.
    2.  It selects the "best" of these (the one with the most robust certified bound)
        and permanently adds it to the solution set.
    3.  It repeats this process with the remaining candidate features until no more
        robust singletons can be found.

    This method is simpler than binary search but may yield a smaller robust set.

    Args:
        model: The Keras model to be analyzed.
        input_sample: The nominal input point.
        lower_bound: The lower bounds of the L-infinity perturbation space.
        upper_bound: The upper bounds of the L-infinity perturbation space.
        xai_indices: A list of feature indices to be excluded from the search.
        free_indices: An initial list of features that are always considered perturbed.
        potential_candidates: An optional subset of features to test.
        channel: The number of channels in the input data.
        data_format: The data format, "channels_first" or "channels_last".
        n_class: The number of output classes.
        decomon_model: An optional, pre-compiled decomon model for efficiency.

    Returns:
        A list of feature indices that were found to be robust through the
        iterative greedy search.
    """

    # step 1: identify singleton that could be free
    best_singleton: list[int]
    other_singleton: list[int]
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
        decomon_model=decomon_model,
    )

    singleton_solutions: list[int] = best_singleton

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
            decomon_model=decomon_model,
        )
        singleton_solutions += best_singleton

    return singleton_solutions
