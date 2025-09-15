import numpy as np


def encode_matrix(n_class:int, groundtruth:int)->np.ndarray:
    """
    Create a matrix of shape (n_class, n_class-1)
    with -1 at the groundtruth row and 1 at each target row (excluding groundtruth).

    Args:
        n_class (int): total number of classes
        groundtruth (int): index of groundtruth label

    Returns:
        np.ndarray: encoded matrix
    """
    gt_vec = np.zeros((n_class,))
    gt_vec[groundtruth]=1
    C_ = np.diag(np.ones((n_class,))) - gt_vec
    return C_[[i for i in range(n_class) if i!=groundtruth]].T

## Abstract Free Set
def get_b(W:np.ndarray, w_u:np.ndarray, b_u:np.ndarray, box:np.ndarray, free_mask:np.ndarray)->np.ndarray:
        # W (batch_size, n_in_wo_channel, n_out)
        # w_u (batch_size, n_in_with_channel, n_out)
        # b_u (batch_size, n_out)
        # box (batch_size, 3, n_in_with_channel)
        # free mask (1, n_in_wo_channel, 1)
        center_:np.ndarray = box[:,2,:,None] # (batch_size, n_in_with_channel, 1)

        b:np.ndarray = np.sum(w_u*center_, 1) + b_u # (batch_size, n_out)
        # add irrelevant features
        b = b + np.sum(W*free_mask, 1) # (batch_size, n_out)

        return b

def get_W(w_u_pos:np.ndarray, w_u_neg:np.ndarray, box:np.ndarray, channel:int=1, data_format:str='channels_first')->np.ndarray:

        n_in_with_channel:int = box.shape[-1]
        n_in_wo_channel:int = int(n_in_with_channel/channel)
        batch_size:int = w_u_pos.shape[0]

        # w_u_(pos, neg) (batch_size, n_in_with_channel, n_out)
        lower_:np.ndarray = box[:,0,:,None] #(batch_size, n_in_with_channel, 1)
        upper_:np.ndarray = box[:,1, :,None] # (batch_size, n_in_with_channel, 1)
        center_:np.ndarray = box[:, 2, :,None] # (batch_size, n_in_with_channel, 1)

        # rescale coefficient and bias
        w:np.ndarray = w_u_pos*(upper_-center_) + w_u_neg*(lower_-center_) # (batch_size, n_in_with_channel, n_out)
        # reshape channel dimension and sum over it
        if data_format=="channels_first":
            w = np.sum(np.reshape(w, (batch_size, channel, n_in_wo_channel, -1)), 1) # (batch_size, n_in_wo_channel, n_out)
        else:
            w = np.sum(np.reshape(w, (batch_size, n_in_wo_channel, channel, -1)), 2) # (batch_size, n_in_wo_channel, n_out)

        return w


def get_xai_mask(n_in:int, xai_indices:list[int])->np.ndarray:
        xai_mask:np.ndarray = np.zeros((n_in,))
        xai_mask[xai_indices]=1 # 1 if potential candidate to add in the xai features
        xai_mask = xai_mask[None, :,None] # (1, n_in, 1)
        return xai_mask

def get_free_mask(n_in:int, free_indices:list[int])->np.ndarray:
        free_mask:np.ndarray = np.zeros((n_in,))
        free_mask[free_indices]=1 # 1 if this index has already been free
        free_mask = free_mask[None, :, None] # (1, n_in, 1)
        return free_mask
