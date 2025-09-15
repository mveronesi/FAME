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
        threshold_k: np.ndarray = np.where(order[p] <= 0)[0][-1]
        coeff_k: list[int] = [
            i for i in i_max[p, :threshold_k] if i not in xai_indices and i not in free_indices
        ][: card_knapsack[p]]
        # cut up to cardinality
        if len(coeff_k):  # potentially empty because only xai_indices and free_indices are
            abstract_free_set[k, np.array(coeff_k)] = 1

    return abstract_free_set
