import numpy as np


def get_greedy(
    index_knapsack: np.ndarray,
    card_knapsack: list[int],
    W: np.ndarray,
    b: np.ndarray,
    xai_indices: list[int],
    free_indices: list[int],
    abstract_free_set: np.ndarray,
) -> np.ndarray:
    """Solves a batch of 0/1 Knapsack-like problems using a greedy heuristic.

    This function provides a fast, suboptimal solution to the problem of selecting
    the largest set of features that can be proven robust. It is used as an
    alternative to a more computationally expensive MILP solver.

    The problem is analogous to a 0/1 Knapsack problem: for each problem instance
    in the batch, we want to select up to `k` (`card_knapsack`) "items" (features)
    to place in our knapsack (the robust set). Each item has a "weight" (its
    impact on the network's output, given by `W`) and we have a maximum "capacity"
    (the budget `b`).

    The greedy strategy works by:
    1. Calculating the "efficiency" of each feature (analogous to a value-to-weight ratio).
    2. Sorting features based on their efficiency (from least to most impactful).
    3. Greedily adding the most efficient features to the solution set until
       either the capacity is reached or the cardinality limit is met.

    Args:
        index_knapsack: An array of indices identifying which problems in the
            batch this function should solve.
        card_knapsack: A list of cardinality limits (`k`), one for each problem
            specified by `index_knapsack`.
        W: The "weights" or "costs" of each feature, derived from an affine
            relaxation. Shape: `(batch_size, n_features, n_outputs)`.
        b: The "budgets" or "capacities" for each problem.
            Shape: `(batch_size, n_outputs)`.
        xai_indices: A list of feature indices to be excluded from selection.
        free_indices: A list of feature indices to be excluded from selection.
        abstract_free_set: The output array (shape: `(N, n_features)`) that
            will be populated with the binary masks of the solutions.

    Returns:
        The `abstract_free_set` array, updated with the greedy solutions for
        the specified knapsack problems.
    """
    # W (batch_size, n_in, n_out)
    # b (batch_size, n_out)
    # |index_knapsack| <= batch_size
    # len(card_knapsack) = len(index_knapsack)
    # xai_indices[i] \in [0.. n_in]
    # free_indices[i] \in [0.. n_in]
    # abstract_free_set (N \ge batch_size, n_in)

    # greedy solution
    # the optimal solution requires a MILP, solve it with a suboptimal greedy approach
    W = -W / b[:, None]  # (batch_size, n_in, n_out)
    bias = -np.ones_like(b)  # (batch_size, n_out)

    i_max: np.ndarray = np.argsort(
        np.max(W, 2)
    )  # consider the worst case impact across all outputs ()
    n_g: int = len(index_knapsack)
    order: np.ndarray = np.max(
        np.cumsum(W[np.arange(n_g)[:, None], i_max], axis=1) + bias[:, None], -1
    )  # (n_g, n_in)
    # threshold:np.ndarray = np.where(order<=0)
    for p, k in enumerate(index_knapsack):
        threshold_k_list: list[int] = np.where(order[p] <= 0)[0]
        if not len(threshold_k_list):
            continue
        threshold_k: np.ndarray = threshold_k_list[-1]
        coeff_k: list[int] = [
            i for i in i_max[p, :threshold_k] if i not in xai_indices and i not in free_indices
        ]
        coeff_k = coeff_k[: card_knapsack[p]]
        # cut up to cardinality
        if len(coeff_k):  # potentially empty because only xai_indices and free_indices are
            abstract_free_set[k, np.sort(coeff_k)] = 1

    return abstract_free_set
