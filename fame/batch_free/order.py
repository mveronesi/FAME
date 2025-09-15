import numpy as np
from fame.batch_free.utils import get_b, get_W


def get_trivial(w_u_trivial:np.ndarray, b_u_trivial:np.ndarray, box_trivial:np.ndarray,
                xai_mask=np.ndarray, free_mask=np.ndarray):
    # we could return all indices up to cardinalities
    # best is to return the highest one from the abstract domain (to facilitate freeing latter one)

    # we keep only indices from card_trivial
    w_u_pos_trivial:np.ndarray = np.maximum(w_u_trivial, 0.) # (|card_trivial|, n_in, n_out)
    w_u_neg_trivial:np.ndarray = np.minimum(w_u_trivial, 0.) # (|card_trivial|, n_in, n_out)

    W_trivial:np.ndarray = get_W(w_u_pos_trivial, w_u_neg_trivial, box_trivial) # (|card_trivial|, n_in, n_out)
    # set xai and free weights to zero (no impact)
    W_trivial = W_trivial*(1-xai_mask)*(1-free_mask)
    b_trivial:np.ndarray = get_b(W_trivial, w_u_trivial, b_u_trivial, box_trivial, free_mask) # (|card_trivial|, n_out)
    W_trivial= -W_trivial/b_trivial[:, None] # (|card_trivial|, 1, n_out)

    return W_trivial
