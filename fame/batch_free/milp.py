from typing import Any, List, Tuple, Union

import cvxpy as cp
import numpy as np


# milp solver for the knapsack solution
def solve_max_cardinality_milp(W: np.ndarray, b: np.ndarray) -> Tuple[int, Union[None, np.ndarray]]:
    """
    Solves for the maximum cardinality of a binary vector x that satisfies Wx + b <= 0.

    Args:
        W (np.ndarray): A numpy array of shape (n_out, n_in).
        b (np.ndarray): A numpy array of shape (n_out,).

    Returns:
        tuple: A tuple containing:
            - the problem
            - max_cardinality (int): The maximum cardinality found.
            - x_solution (np.ndarray): The binary vector x that achieves this cardinality.
            Returns (None, None) if the problem is infeasible.
    """
    # Get the dimensions from the input arrays
    if W.ndim == 1:
        W = W.reshape(1, -1)
    if b.ndim == 0:
        b = b.reshape(
            1,
        )

    n_in: int
    _, n_in = W.shape

    # 1. Define the MILP variable
    # x is a vector of size n_in with boolean (0 or 1) entries.
    x: Any = cp.Variable(n_in, boolean=True)

    # 2. Define the objective function
    # We want to maximize the sum of the elements of x, which is its cardinality.
    objective: Any = cp.Maximize(cp.sum(x))

    # 3. Define the constraints
    # The constraint W*x + b <= 0 must hold for all n_out dimensions.
    constraints: List[Any] = [W @ x + b <= 0]

    # 4. Formulate and solve the problem
    problem: cp.Problem = cp.Problem(objective, constraints)
    # verbose=True can be added for solver output
    problem.solve()

    # 5. Check the result and return the solution
    if problem.status in ["infeasible", "unbounded"]:
        print("The problem is infeasible. No solution exists.")
        return 0, None

    max_cardinality: int = int(problem.value)
    # x.value might contain small floating point errors, so we round and cast
    x_solution: np.ndarray = np.round(x.value).astype(int)

    return max_cardinality, x_solution


def get_milp_sample(
    W: np.ndarray, b: np.ndarray, xai_indices: list[int], free_indices: list[int], cardinality: int
) -> list[int]:
    card_opt, solution_opt = solve_max_cardinality_milp(W.T, b)
    if card_opt:
        coeff_opt = [
            i
            for i in np.where(solution_opt == 1)[0]
            if i not in xai_indices and i not in free_indices
        ][:cardinality]
    else:
        coeff_opt = []

    return coeff_opt


def get_milp(
    index_knapsack: np.ndarray,
    card_knapsack: list[int],
    W: np.ndarray,
    b: np.ndarray,
    xai_indices: list[int],
    free_indices: list[int],
    abstract_free_set: np.ndarray,
) -> np.ndarray:
    for p, k in enumerate(index_knapsack):
        coeff_k = get_milp_sample(
            W=W[p],
            b=b[p],
            xai_indices=xai_indices,
            free_indices=free_indices,
            cardinality=card_knapsack[p],
        )
        if len(coeff_k):
            abstract_free_set[k, np.array(coeff_k)] = 1

    return abstract_free_set
