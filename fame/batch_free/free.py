from typing import Tuple

import keras
import numpy as np
from decomon import clone
from decomon.perturbation_domain import PerturbationDomain
from fame.abstract_domain.abstract import (
    get_abstract_model as get_abstract_model_singleton,
)
from fame.abstract_domain.cardinality_domain import XAIDomain
from fame.batch_free.utils import encode_matrix
from keras import KerasTensor as Tensor
from keras.layers import Input

from .greedy import get_greedy
from .milp import get_milp
from .singleton import free_with_binary_search, free_with_singleton_search
from .utils import get_b, get_free_mask, get_W, get_xai_mask


def get_features_batch(
    model: keras.models.Model,
    gt_label: int,
    input_sample: Tensor,
    lower_bound_input: Tensor,
    upper_bound_input: Tensor,
    xai_indices: list[int],
    free_indices: list[int],
    cardinality: np.ndarray,
    channel: int = 1,
    data_format: str = "channels_first",
    n_class: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Performs a batched abstract interpretation pass over a hybrid L-inf/L0 domain.

    This function is a low-level wrapper around `decomon` that analyzes a complex
    perturbation space in a single, batched forward pass. It constructs a
    specialized `XAIDomain` which models a hybrid perturbation where:
    1. A set of `free_indices` are always perturbed within an L-infinity ball.
    2. Up to `k` features (defined by `cardinality`) from the `xai_indices`
       pool can also be perturbed (L0-norm constraint).

    It then performs backward bound propagation (CROWN) to extract both the final
    concrete upper bounds and the parameters of the final affine relaxation.

    Args:
        model: The Keras model to analyze.
        gt_label: The ground-truth class label.
        input_sample: The nominal input point.
        lower_bound_input: The lower bounds of the L-infinity perturbation.
        upper_bound_input: The upper bounds of the L-infinity perturbation.
        xai_indices: A list of feature indices subject to the L0 constraint.
        free_indices: A list of feature indices always perturbed under L-infinity.
        cardinality: A numpy array where each element specifies the L0 budget (`k`)
            for the corresponding item in the batch.
        channel: The number of channels in the input data.
        data_format: The data format, "channels_first" or "channels_last".
        n_class: The number of output classes of the model.

    Returns:
        A tuple of four numpy arrays:
        - `w_u`: The weights of the final affine upper bound with respect to the
          input features, shape `(batch, n_features, n_outputs)`.
        - `b_u`: The bias of the final affine upper bound, shape `(batch, n_outputs)`.
        - `upper`: The concrete (IBP) upper bounds on the logit differences.
        - `box`: The input domain tensor `[lower, upper, center]` used for analysis.
    """

    n_in_with_channel: int = input_sample.shape[-1]
    n_in_wo_channel: int = int(n_in_with_channel / channel)
    batch_size: int = len(cardinality)

    # lower_bound/upper_bound/input_sample.shape = (n_in,)
    lower_bound_batch: np.ndarray = np.repeat(
        np.copy(lower_bound_input[None]) + 0.0, repeats=batch_size, axis=0
    )  # (batch_size, n_in_with_channel)
    upper_bound_batch: np.ndarray = np.repeat(
        np.copy(upper_bound_input[None]) + 0.0, repeats=batch_size, axis=0
    )  # (batch_size, n_in_with_channel)
    input_sample_batch: np.ndarray = np.repeat(
        np.copy(input_sample[None] + 0.0), repeats=batch_size, axis=0
    )  # (batch_size, n_in_with_channel)

    # step 1: build your PerturbationDomain
    xai_perturbation_domain: PerturbationDomain = XAIDomain(
        xai_indices=xai_indices,
        free_indices=free_indices,
        cardinalities=cardinality,
        n_dim=n_in_wo_channel,
        channel=channel,
        data_format=data_format,
    )
    box: np.ndarray = np.concatenate(
        [lower_bound_batch[:, None], upper_bound_batch[:, None], input_sample_batch[:, None]], 1
    )  # (batch_size, 3, n_in_with_channel)

    # build your input domain
    # encode matrix C
    C: Tensor = Input((n_class, n_class - 1))
    C_gt: np.ndarray = np.repeat(
        encode_matrix(n_class=n_class, groundtruth=gt_label)[None], repeats=batch_size, axis=0
    )  # (batch_size, n_class, n_class-1)

    decomon_model: keras.model.Model = clone(
        model,
        perturbation_domain=xai_perturbation_domain,
        final_affine=True,
        final_ibp=True,
        final_lower=False,
        backward_bounds=[C],
    )  # return only upper bound

    w_u: np.ndarray
    b_u: np.ndarray
    upper: np.ndarray
    w_u, b_u, upper = decomon_model.predict([box, C_gt], verbose=0, batch_size=len(box))

    return w_u, b_u, upper, box


def free_at_once_k_features(
    model: keras.models.Model,
    gt_label: int,
    input_sample: np.ndarray,
    lower_bound_input: np.ndarray,
    upper_bound_input: np.ndarray,
    xai_indices: list[int] = [],
    free_indices: list[int] = [],
    cardinality: np.ndarray = None,
    channel: int = 1,
    data_format: str = "channels_first",
    n_class: int = 10,
    method: str = "greedy",
    verbose: int = 0,
) -> np.ndarray:
    """Finds the largest safe set of features, given cardinality constraints.

    This function attempts to find the largest set of features that can be added
    to the "robust set" (i.e., allowed to be perturbed) without violating the
    model's overall robustness. It solves this for a batch of different L0-norm
    cardinality constraints (`k`).

    The process involves:
    1.  Using abstract interpretation (`get_features_batch`) to obtain an affine
        approximation of the network's output logits.
    2.  Formulating a 0-1 Knapsack-like optimization problem from this approximation.
        The goal is to select the maximum number of features to "free" while
        ensuring the certified upper bound on logit differences remains non-positive.
    3.  Solving this problem using either an exact MILP solver or a fast greedy heuristic.

    Args:
        model: The Keras model to analyze.
        gt_label: The ground-truth class label.
        input_sample: The nominal input point.
        lower_bound_input: The lower bounds of the L-infinity perturbation.
        upper_bound_input: The upper bounds of the L-infinity perturbation.
        xai_indices: Candidate features for the knapsack problem.
        free_indices: Features that are always considered robust/perturbed.
        cardinality: A numpy array where each element specifies the L0 budget (`k`)
            for the corresponding item in the batch.
        channel: The number of channels in the input data.
        data_format: The data format, "channels_first" or "channels_last".
        n_class: The number of output classes.
        method: The optimization method to use: "milp" for an exact solution or
            "greedy" for a fast approximation.
        verbose: Verbosity level.

    Returns:
        A numpy array of shape `(batch_size, n_features)`, where each row is a
        binary mask indicating the set of features that can be safely made robust
        for the corresponding cardinality constraint.
    """
    lower_bound: np.ndarray = np.copy(lower_bound_input) + 0.0
    upper_bound: np.ndarray = np.copy(upper_bound_input) + 0.0

    n_in_with_channel: int = input_sample.shape[-1]
    n_in_wo_channel: int = int(n_in_with_channel / channel)  # shortcut n_in

    input_: np.ndarray
    if data_format == "channels_first":
        lower_bound = np.reshape(
            lower_bound, (channel, n_in_wo_channel)
        )  # (channel, n_in_wo_channel)
        upper_bound = np.reshape(upper_bound, (channel, n_in_wo_channel))
        input_: np.ndarray = np.reshape(np.copy(input_sample) + 0.0, (channel, n_in_wo_channel))
    else:
        lower_bound = np.reshape(
            lower_bound, (n_in_wo_channel, channel)
        )  # (n_in_wo_channel, channel)
        upper_bound = np.reshape(upper_bound, (n_in_wo_channel, channel))
        input_: np.ndarray = np.reshape(np.copy(input_sample) + 0.0, (n_in_wo_channel, channel))

    # freeze xai features to the nominal value
    if len(xai_indices):
        if data_format == "channels_first":
            lower_bound[:, xai_indices] = input_[:, xai_indices]
            upper_bound[:, xai_indices] = input_[:, xai_indices]
        else:
            lower_bound[xai_indices, :] = input_[xai_indices, :]
            upper_bound[xai_indices, :] = input_[xai_indices, :]

    # reflatten
    lower_bound = np.reshape(lower_bound, (n_in_with_channel,))
    upper_bound = np.reshape(upper_bound, (n_in_with_channel,))

    batch_size: int = len(cardinality)

    w_u: np.ndarray
    b_u: np.ndarray
    box: np.ndarray
    w_u, b_u, upper, box = get_features_batch(
        model=model,
        gt_label=gt_label,
        input_sample=input_sample,
        lower_bound_input=lower_bound,
        upper_bound_input=upper_bound,
        xai_indices=xai_indices,
        free_indices=free_indices,
        cardinality=cardinality,
        channel=channel,
        data_format=data_format,
        n_class=n_class,
    )

    # w_u (batch_size, n_in_with_channel, n_class-1)
    # b_u (batch_size, n_class-1)
    # upper (batch_size, n_class-1)
    # box (batch_size, 3, n_in_with_channel)
    abstract_free_set: np.ndarray = np.zeros(
        (batch_size, n_in_wo_channel)
    )  # fill with 1 if in the solution

    if verbose:
        print("upper", upper)
        print("b", b_u)

    # TO FINISH
    w_u_pos: np.ndarray = np.maximum(w_u, 0.0)  # (batch_size, n_in_with_channel, n_class-1)
    w_u_neg: np.ndarray = np.minimum(w_u, 0.0)  # (batch_size, n_in_with_channel, n_class-1)

    card_index: np.ndarray = np.array([i for i in np.arange(batch_size) if np.max(upper[i]) > 0])
    # trivial case, return the whole set given traversal order and cardinality constraints
    card_trivial: np.ndarray = np.array([i for i in np.arange(batch_size) if np.max(upper[i]) <= 0])

    xai_mask: np.ndarray = get_xai_mask(n_in_wo_channel, xai_indices)  # (1, n_in_wo_channel, 1)
    free_mask: np.ndarray = get_free_mask(n_in_wo_channel, free_indices)  # (1, n_in_wo_channel, 1)

    if len(card_trivial):
        # we could return all indices up to cardinalities
        # best is to return the highest one from the abstract domain (to facilitate freeing latter one)

        # we keep only indices from card_trivial
        w_u_trivial: np.ndarray = w_u[card_trivial]  # (|card_trivial|, n_in_with_channel, n_out)
        b_u_trivial: np.ndarray = b_u[card_trivial]  # (|card_trivial|, n_out)
        w_u_pos_trivial: np.ndarray = w_u_pos[
            card_trivial
        ]  # (|card_trivial|, n_in_with_channel, n_out)
        w_u_neg_trivial: np.ndarray = w_u_neg[
            card_trivial
        ]  # (|card_trivial|, n_in_with_channel, n_out)
        box_trivial: np.ndarray = box[card_trivial]  # (|card_trivial|, n_in_with_channel)
        W_trivial: np.ndarray = get_W(
            w_u_pos_trivial, w_u_neg_trivial, box_trivial, channel=channel, data_format=data_format
        )  # (|card_trivial|, n_in_wo_channel, n_out)

        # set xai and free weights to zero (no impact)
        W_trivial = W_trivial * (1 - xai_mask) * (1 - free_mask)
        b_trivial: np.ndarray = get_b(
            W_trivial, w_u_trivial, b_u_trivial, box_trivial, free_mask
        )  # (|card_trivial|, n_out)
        W_trivial = -W_trivial / b_trivial[:, None]  # (|card_trivial|, 1, n_out)

        # consider the least case impact across all outputs
        i_max_trivial: np.ndarray = np.argsort(
            np.max(W_trivial, 2)
        )  # (|card_trivial|, n_in_wo_channel)
        for j, k in enumerate(card_trivial):
            indices_k: np.ndarray = np.array(
                [i for i in i_max_trivial[j] if i not in xai_indices and i not in free_indices][
                    : cardinality[k]
                ]
            )
            # indices_k = np.array([i for i in range(n_in) if i not in xai_indices and i not in free_indices][:cardinality[k]])
            abstract_free_set[k, indices_k] = 1

    if len(card_index) == 0:
        # only trivial solutions
        return abstract_free_set

    # kept only indices from card_index
    w_u_keep: np.ndarray = w_u[card_index]  # (|card_index|, n_in_with_channel, n_class-1)
    w_u_pos_keep: np.ndarray = w_u_pos[card_index]  # (|card_index|, n_in_with_channel, n_class-1)
    w_u_neg_keep: np.ndarray = w_u_neg[card_index]  # (|card_index|, n_in_with_channel, n_class-1)
    b_u_keep: np.ndarray = b_u[card_index]  # (|card_index|, n_class-1)
    box_keep: np.ndarray = box[card_index]  # (|card_index|, n_in_with_channel)

    W: np.ndarray = get_W(
        w_u_pos_keep, w_u_neg_keep, box_keep, channel=channel, data_format=data_format
    )  # (|card_ind|, n_in_wo_channel, n_out)
    b: np.ndarray = get_b(W, w_u_keep, b_u_keep, box_keep, free_mask)  # (|card_ind|, n_out)

    # set xai and free weights to zero (no impact)
    W = W * (1 - xai_mask) * (1 - free_mask)  # (|card_ind|, n_in_wo_channel, n_out)

    # index_irrelevant:list[int] = [card_index[i] for i in range(len(card_index)) if np.max(b[i]) >0]
    index_knapsack: list[int] = np.array(
        [p for (p, i) in enumerate(card_index) if np.max(b[p]) <= 0]
    )  # (b_g,) b_g <= |card_ind|
    card_knapsack: list[int] = np.array(
        [cardinality[i] for (p, i) in enumerate(card_index) if np.max(b[p]) <= 0]
    )

    if len(index_knapsack) == 0:
        # abstract set of irrelevant features is empty
        return abstract_free_set

    W = W[index_knapsack]  # (b_g, n_in_wo_channel, n_out)
    b = b[index_knapsack]  # (b_g, n_out)

    if method == "milp":
        abstract_free_set = get_milp(
            card_index[index_knapsack],
            card_knapsack,
            W,
            b,
            xai_indices,
            free_indices,
            abstract_free_set,
        )
    elif method == "greedy":
        abstract_free_set = get_greedy(
            card_index[index_knapsack],
            card_knapsack,
            W,
            b,
            xai_indices,
            free_indices,
            abstract_free_set,
        )

    else:
        raise ValueError("method {} is unknown".format(method))

    return abstract_free_set


def free_iteratively_k_features(
    model: keras.models.Model,
    gt_label: int,
    input_sample: np.array,
    eps: float = 0.0,
    xai_indices: list[int] = [],
    free_indices: list[int] = [],
    channel: int = 1,
    data_format: str = "channels_first",
    n_class: int = 10,
    method: str = "greedy",
    refining_domain: bool = True,
    verbose: int = 0,
) -> tuple[list[int], list[int]]:
    """Iteratively finds the largest possible set of robust features for a given input.

    This function implements a high-level iterative algorithm to discover the
    largest set of features that can be perturbed (the "free set") without
    affecting the model's prediction. It provides a formal under-approximation
    of the minimal set of features required to explain a prediction.

    The process involves two main phases:
    1.  **Iterative Set Expansion**: If `refining_domain` is True, it repeatedly
        calls `free_at_once_k_features` to find large groups of features that
        can be safely added to the robust set. After each successful find, it
        expands `free_indices` and repeats the search on the smaller remaining
        pool of features until no more groups can be found.
    2.  **Singleton Refinement**: After the group expansion phase, it performs a
        final, fine-grained search (`free_with_binary_search`) to check if any
        of the remaining individual features can also be safely added to the
        robust set.

    Args:
        model: The Keras model to analyze.
        gt_label: The ground-truth class label.
        input_sample: The nominal input point.
        eps: The radius of the $L_\infty$ perturbation.
        xai_indices: A list of feature indices to be explained.
        free_indices: An initial list of features already known to be robust.
        channel: The number of channels in the input data.
        data_format: The data format, "channels_first" or "channels_last".
        n_class: The number of output classes.
        method: The optimization method ("milp" or "greedy") for the sub-problem.
        refining_domain: If True, iteratively expands the robust set until a
            fixed point is reached.
        verbose: Verbosity level.

    Returns:
        A tuple of two lists of feature indices:
        - The first list is the final, largest set of "free" (robust) features found.
        - The second list contains the remaining features that could not be
          proven robust by this method.
    """
    n_in_with_channel: int = input_sample.shape[-1]
    n_in_wo_channel: int = int(n_in_with_channel / channel)
    lower_bound_input: np.ndarray = np.maximum(np.copy(input_sample) - eps, 0 * input_sample)
    upper_bound_input: np.ndarray = np.minimum(np.copy(input_sample) + eps, 0 * input_sample + 1)

    cardinality: np.ndarray = np.array([i for i in range(1, n_in_wo_channel - len(free_indices))])
    abstract_set = np.zeros((1,))  # temporary

    abstract_set: np.ndarray = free_at_once_k_features(
        model=model,
        gt_label=gt_label,
        input_sample=np.copy(input_sample) + 0.0,
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

    if refining_domain:
        while (
            abstract_set.sum(-1).max() != 0
        ):  # we have found new input features to free among remaining indices
            i_solution = np.argmax(
                np.sum(abstract_set, -1)
            )  # find the cardinality that propose the largest cardinality to free

            free_indices += [
                i
                for (i, k) in enumerate(abstract_set[i_solution])
                if k == 1 and not i in free_indices
            ]

            # update cardinality
            cardinality = np.array([i for i in range(1, n_in_wo_channel + 1 - len(free_indices))])

            if not len(cardinality):
                raise ValueError(
                    "cardinality should not be empty as the local region is not robust unless the local region is robust"
                )

            # overwrite the bounds in case they have been corrupted
            lower_bound_input = np.maximum(input_sample - eps, 0 * input_sample)
            upper_bound_input = np.minimum(input_sample + eps, 0 * input_sample + 1)

            abstract_set = free_at_once_k_features(
                model=model,
                gt_label=gt_label,
                input_sample=np.copy(input_sample) + 0.0,
                lower_bound_input=lower_bound_input,
                upper_bound_input=upper_bound_input,
                xai_indices=xai_indices,
                free_indices=free_indices,
                cardinality=cardinality,
                channel=channel,
                data_format=data_format,
                n_class=n_class,
                method=method,
                verbose=0,
            )

    # we consider the tightest abstract domain at our disposal: singleton + set of current free features
    # while we find one singleton (we add the one with the least impact according to abstract bound)
    # finish with singleton search
    decomon_singleton = get_abstract_model_singleton(model=model, n_class=n_class)
    lower_bound_input = np.maximum(input_sample - eps, 0 * input_sample)
    upper_bound_input = np.minimum(input_sample + eps, 0 * input_sample + 1)
    singleton_free_index: list
    singleton_free_index = free_with_binary_search(
        model=model,
        input_sample=np.copy(input_sample) + 0.0,
        lower_bound=lower_bound_input,
        upper_bound=upper_bound_input,
        free_indices=free_indices,
        potential_candidates=None,  #
        xai_indices=xai_indices,
        decomon_model=decomon_singleton,
        channel=channel,
        data_format=data_format,
        n_class=n_class,
    )

    return free_indices, singleton_free_index
