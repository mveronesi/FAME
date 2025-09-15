
from typing import Any, List, Tuple, Union

import keras
import keras.ops as K
import numpy as np
from keras import KerasTensor as Tensor


def get_upper_box_l0(x_min: Tensor, x_max: Tensor, x_center:Tensor,
                     w: Tensor, b: Tensor, mask_xai:np.ndarray, mask_free:np.ndarray,
                     channel:int, data_format:str,
                     cardinality:Union[int, List[int]], **kwargs:Any) -> Tensor:

    if data_format=="channels_last":
        raise NotImplementedError()

    missing_batchsize:bool = 'missing_batchsize' in kwargs and kwargs['missing_batchsize']
    if missing_batchsize:
        w=w[None]
        b=b[None]

    # split into positive and negative components
    z_value:Tensor = K.cast(0.0, dtype=x_min.dtype)
    n_h: list[int]= len(w.shape[2:]) # output shape of the layer
    n_in: int = w.shape[1]//channel # number of input features (without the channel dimension)

    # assuming channels_first
    if data_format=="channels_first":
        w = K.reshape(w, (-1, channel, n_in)+w.shape[2:]) # (None, c, n_in, n_h)
    else:
        w = K.reshape(w, (-1, n_in, channel)+w.shape[2:]) # (None, n_in, c, n_h)

    w_pos:Tensor = K.relu(w) # (None, c, n_in, n_h) assuming data_format == channels_first
    w_neg:Tensor = w - w_pos # (None, c, n_in, n_h)

    # expand dimension of x_min and x_max (None, channel, n_in) assuming channels_first
    axis_channel:int
    if data_format=="channels_first":
        x_min_out:Tensor = K.reshape(x_min, [-1, channel, n_in]+[1]*n_h) # x_min_out, x_max_out (None, c, n_in, 1..)
        x_max_out:Tensor = K.reshape(x_max, [-1, channel, n_in]+[1]*n_h)
        x_center_out:Tensor = K.reshape(x_center, [-1, channel, n_in]+[1]*n_h)
        axis_channel = 1
        # get the nominal value (warning clipping) (None, c, n_in, 1..)
    else:
        x_min_out:Tensor = K.reshape(x_min, [-1, n_in, channel]+[1]*n_h) # x_min_out, x_max_out (None, n_in, c, 1..)
        x_max_out:Tensor = K.reshape(x_max, [-1, n_in, channel]+[1]*n_h)
        x_center_out:Tensor = K.reshape(x_center, [-1, n_in, channel]+[1]*n_h)
        # get the nominal value (warning clipping) (None, n_in, c, 1..)
        axis_channel = 2

    mask_xai_out:Tensor = K.reshape(mask_xai, [-1, n_in]+[1]*n_h) # mask_xai_out, mask_free_out (None, n_in, 1..)
    mask_free_out:Tensor = K.reshape(mask_free, [-1, n_in]+[1]*n_h)


    scoring_samples:Tensor = K.sum(w_pos * (x_max_out-x_center_out) +\
                                    w_neg * (x_min_out-x_center_out), axis_channel) # (None, n_in, 1..)


    # get the scoring samples for each dimension (None, n_in, n_h):

    # xai and free should have really low value so not to be considered
    # scoring samples is always positive so we can set the scoring samples to 0 for xai and free if we don't want to select it
    # xai because the indices are already selected and thus the dimension will be set to the nominal value
    # free because the indices are always freed
    scoring_samples_wo_free:Tensor = scoring_samples*(1-mask_xai_out)*(1-mask_free_out) # (None, n_in, n_h)

    # select cardinality samples from 1..n_in not in xai or free to maximize scoring_samples
    # keras.ops.sort is in ascending order. We sort it along axis n_in and take the last values
    # cardinality does not take into account xai indices and free indices that are set to 0

    # ascending order
    if isinstance(cardinality, int):
        # ascending order
        scoring_rank:Tensor = -keras.ops.sort(-scoring_samples_wo_free, axis=1) # (None, n_in, n_h)
        threshold:Tensor = scoring_rank[:, cardinality-1][:,None] # (None, 1, n_h..)
    else:
        batch_size:int = len(cardinality)
        threshold:Tensor = -keras.ops.sort(-scoring_samples_wo_free, axis=1)[np.arange(batch_size), cardinality-1][:,None]
        # (None, 1, n_h..)

    # keep only scoring_samples lower than threshold
    final_score_mask:Tensor = K.cast(threshold<= scoring_samples_wo_free, 'int')
    final_score:Tensor = K.sum(scoring_samples_wo_free*final_score_mask, axis=1) # (None, n_h, ...)


    bias:Tensor = b + K.sum(K.sum(w*x_center_out, axis=axis_channel), axis=1) # (None, n_h...) sum over channel
    # update bias with free indices !
    free_scoring_samples:Tensor = K.sum(scoring_samples*mask_free_out, 1)
    bias = bias + free_scoring_samples

    return final_score+bias


def get_lower_box_l0(x_min: Tensor, x_max: Tensor, x_center:Tensor,
                     w: Tensor, b: Tensor,
                     mask_xai:np.ndarray, mask_free:np.ndarray,
                     channel:int, data_format:str,
                     cardinality:Union[int, List[int]], **kwargs:Any) -> Tensor:


    return -get_upper_box_l0(x_min=x_min,
                             x_max=x_max,
                             x_center=x_center,
                             w= -w,
                             b=-b,
                             mask_xai=mask_xai,
                             mask_free=mask_free,
                             channel=channel,
                             data_format=data_format,
                             cardinality=cardinality,
                             **kwargs)
