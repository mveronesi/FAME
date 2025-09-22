from typing import Tuple, Union

import keras
import numpy as np
from decomon import clone
from keras import KerasTensor as Tensor
from keras.layers import Input

from ..batch_free.utils import encode_matrix

from typing import Union


def get_abstract_model(
    model: keras.models.Model, n_class: int = 10, final_affine: bool = False, final_ibp: bool = True
) -> keras.models.Model:
    C: Tensor = Input((n_class, n_class - 1))
    # (batch_size, n_class, n_class-1)

    decomon_model: keras.model.Model = clone(
        model,
        final_affine=final_affine,
        final_ibp=final_ibp,
        final_lower=False,
        backward_bounds=[C],
    )  # return only upper bound

    return decomon_model


def get_abstract_output_domain_singleton(
    model: keras.models.Model,
    input_sample: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    xai_indices: list[int],
    free_indices: list[int],
    remaining_indices:Union[list[int], None]=None, # a subset of remaining dimensions or None
    channel: int = 1,
    data_format: str = "channels_first",
    n_class: int = 10,
    decomon_model: keras.models.Model = None,
) -> np.ndarray:
    n_in_with_channel: int = input_sample.shape[-1]
    n_in_wo_channel: int = int(n_in_with_channel / channel)

    if data_format == "channels_first":
        # (channel, n_in_wo_channel)
        input_sample_c: np.ndarray = np.reshape(input_sample, (channel, n_in_wo_channel))
        lower_bound_c: np.ndarray = np.reshape(lower_bound, (channel, n_in_wo_channel))
        upper_bound_c: np.ndarray = np.reshape(upper_bound, (channel, n_in_wo_channel))
    else:  # channels_last
        # (n_in_wo_channel, channel)
        input_sample_c: np.ndarray = np.reshape(input_sample, (n_in_wo_channel, channel))
        lower_bound_c: np.ndarray = np.reshape(lower_bound, (n_in_wo_channel, channel))
        upper_bound_c: np.ndarray = np.reshape(upper_bound, (n_in_wo_channel, channel))

    batch_size:int
    if remaining_indices is None:
        # consider every remaining dimensions
        batch_size= (
            n_in_wo_channel - len(xai_indices) - len(free_indices)
        )  # number of remaining indices
    else:
        # remaining_indices has been set previously by the pipeline
        batch_size = len(remaining_indices)

    gt_label: int = model.predict(input_sample[None], verbose=0).argmax(-1)[0]

    # repeat subdomains
    # freeze xai features to the nominal value
    lower_bound_np: np.ndarray = (
        np.copy(input_sample_c) + 0.0
    )  # (channel, n_in_wo_channel) or (n_in_wo_channel, channel)
    upper_bound_np: np.ndarray = np.copy(input_sample_c) + 0.0
    if len(free_indices):
        if data_format == "channels_first":
            lower_bound_np[:, free_indices] = lower_bound_c[
                :, free_indices
            ]  # open the free dimension
            upper_bound_np[:, free_indices] = upper_bound_c[:, free_indices]
        else:
            lower_bound_np[free_indices, :] = lower_bound_c[
                free_indices, :
            ]  # open the free dimension
            upper_bound_np[free_indices, :] = upper_bound_c[free_indices, :]

    # repeat
    lower_bound_batch: np.ndarray = np.repeat(
        lower_bound_np[None], repeats=batch_size, axis=0
    )  # (batch_size, channel, n_in_wo_channel)
    upper_bound_batch: np.ndarray = np.repeat(
        upper_bound_np[None], repeats=batch_size, axis=0
    )  # (batch_size, channel, n_in_wo_channel)

    # expand one dimension in the set of remaining features
    if remaining_indices is None:
        remaining_indices = [i for i in range(n_in_wo_channel) if i not in xai_indices + free_indices]
    
    assert (
            len(remaining_indices) == batch_size
    ), "Value Error remaining indices length should match batch_size"
    
    for i, j in enumerate(remaining_indices):
        if data_format == "channels_first":
            lower_bound_batch[i, :, j] = lower_bound_c[:, j]
            upper_bound_batch[i, :, j] = upper_bound_c[:, j]
        else:
            try:
                lower_bound_batch[i, j, :] = lower_bound_c[j, :]
                upper_bound_batch[i, j, :] = upper_bound_c[j, :]
            except:
                import pdb; pdb.set_trace()

    # flatten lower_bound_batch and upper_bound_batch
    lower_bound_batch = np.reshape(
        lower_bound_batch, (-1, 1, n_in_with_channel)
    )  # (batch_size, 1, n_in_with_channel)
    upper_bound_batch = np.reshape(
        upper_bound_batch, (-1, 1, n_in_with_channel)
    )  # (batch_size, 1, n_in_with_channel)
    box: np.ndarray = np.concatenate(
        [lower_bound_batch, upper_bound_batch], 1
    )  # (batch_size, 2, n_in_with_channel)

    # build your input domain
    # encode matrix C
    C_gt: np.ndarray = np.repeat(
        encode_matrix(n_class=n_class, groundtruth=gt_label)[None], repeats=batch_size, axis=0
    )
    # (batch_size, n_class, n_class-1)

    if decomon_model is None:
        C: Tensor = Input((n_class, n_class - 1))

        decomon_model: keras.model.Model = clone(
            model, final_affine=False, final_ibp=True, final_lower=False, backward_bounds=[C]
        )  # return only upper bound
        
    upper: np.ndarray = decomon_model.predict([box, C_gt], verbose=0)

    return upper

def get_abstract_output_domain(
    model: keras.models.Model,
    input_sample: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    xai_indices: list[int],
    free_indices: list[int],
    channel: int = 1,
    data_format: str = "channels_first",
    n_class: int = 10,
    decomon_model: keras.models.Model = None,
) -> np.ndarray:
    n_in_with_channel: int = input_sample.shape[-1]
    n_in_wo_channel: int = int(n_in_with_channel / channel)

    if data_format == "channels_first":
        # (channel, n_in_wo_channel)
        input_sample_c: np.ndarray = np.reshape(input_sample, (channel, n_in_wo_channel))
        lower_bound_c: np.ndarray = np.reshape(input_sample, (channel, n_in_wo_channel))
        upper_bound_c: np.ndarray = np.reshape(upper_bound, (channel, n_in_wo_channel))
    else:  # channels_last
        # (n_in_wo_channel, channel)
        input_sample_c: np.ndarray = np.reshape(input_sample, (n_in_wo_channel, channel))
        lower_bound_c: np.ndarray = np.reshape(lower_bound, (n_in_wo_channel, channel))
        upper_bound_c: np.ndarray = np.reshape(upper_bound, (n_in_wo_channel, channel))

    gt_label: int = model.predict(input_sample[None], verbose=0).argmax(-1)[0]

    # repeat subdomains
    # freeze xai features to the nominal value
    lower_bound_np: np.ndarray = (
        np.copy(input_sample_c) + 0.0
    )  # (channel, n_in_wo_channel) or (n_in_wo_channel, channel)
    upper_bound_np: np.ndarray = np.copy(input_sample_c) + 0.0
    if len(free_indices):
        if data_format == "channels_first":
            lower_bound_np[:, free_indices] = lower_bound_c[
                :, free_indices
            ]  # open the free dimension
            upper_bound_np[:, free_indices] = upper_bound_c[:, free_indices]
        else:
            lower_bound_np[free_indices, :] = lower_bound_c[
                free_indices, :
            ]  # open the free dimension
            upper_bound_np[free_indices, :] = upper_bound_c[free_indices, :]

    # flatten lower_bound_batch and upper_bound_batch
    lower_bound_batch = np.reshape(
        lower_bound_np, (1, 1, n_in_with_channel)
    )  # (batch_size, 1, n_in_with_channel)
    upper_bound_batch = np.reshape(
        upper_bound_np, (1, 1, n_in_with_channel)
    )  # (batch_size, 1, n_in_with_channel)
    box: np.ndarray = np.concatenate(
        [lower_bound_batch, upper_bound_batch], 1
    )  # (batch_size, 2, n_in_with_channel)

    # build your input domain
    # encode matrix C
    C_gt: np.ndarray = encode_matrix(n_class=n_class, groundtruth=gt_label)[None]# (1, n_class, n_class-1)

    if decomon_model is None:
        C: Tensor = Input((n_class, n_class - 1))

        decomon_model: keras.model.Model = clone(
            model, final_affine=False, final_ibp=True, final_lower=False, backward_bounds=[C]
        )  # return only upper bound
        
    upper: np.ndarray = decomon_model.predict([box, C_gt], verbose=0)

    return upper
