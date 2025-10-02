from typing import Any, List, Tuple, Union

import keras
import keras.ops as K
import numpy as np
from decomon.perturbation_domain import get_upper_box
from fame.abstract_domain.abstract import get_abstract_output_domain
from keras import KerasTensor as Tensor


def get_upper_box_l0(
    x_min: Tensor,
    x_max: Tensor,
    x_center: Tensor,
    w: Tensor,
    b: Tensor,
    mask_xai: np.ndarray,
    mask_free: np.ndarray,
    channel: int,
    data_format: str,
    cardinality: Union[int, List[int]],
    **kwargs: Any,
) -> Tensor:
    """Computes the upper bound of a linear operation over a hybrid L-infinity/L0 domain.

    This function implements an abstract transformer for a linear layer (`w*x + b`)
    over a complex perturbation domain. The domain consists of three types of
    features:
    1.  "Free" features (`mask_free`), which are always perturbed within an
        L-infinity box defined by `x_min` and `x_max`.
    2.  "XAI" candidate features (`mask_xai`) which remain at their nominal `x_center` value.
    3. All other features (remaining features) from which up to `cardinality`
        features can be chosen to be perturbed to maximize the output (L0-norm constraint).

    The method works by first calculating the maximum possible positive contribution
    ("score") each feature could make to the output. It then greedily selects
    the top `cardinality` candidate features with the highest scores and sums
    their contributions. The final bound is the sum of the output at the nominal
    center, the contribution from the "free" features, and the contribution from
    the selected top-k "XAI" features.

    Args:
        x_min: Tensor defining the lower bounds of the L-infinity component of the domain.
        x_max: Tensor defining the upper bounds of the L-infinity component of the domain.
        x_center: Tensor for the nominal center of the perturbation domain.
        w: Weight tensor of the linear layer.
        b: Bias tensor of the linear layer.
        mask_xai: A binary mask identifying features that are candidates for L0
            perturbation.
        mask_free: A binary mask identifying features that are always perturbed
            under the L-infinity norm.
        channel: The number of channels in the input data.
        data_format: The data format, either "channels_first" or "channels_last".
        cardinality: The L0 norm budget. The maximum number of features from the
            `xai_mask` pool to perturb. Can be an integer for the entire batch or
            a list of integers for per-sample budgets.
        **kwargs: Additional keyword arguments, used internally.

    Returns:
        A tensor representing the computed upper bound of the linear operation.
    """

    missing_batchsize: bool = "missing_batchsize" in kwargs and kwargs["missing_batchsize"]

    if missing_batchsize:
        w = w[None]
        b = b[None]

    # split into positive and negative components
    n_h: list[int] = len(w.shape[2:])  # output shape of the layer
    n_in: int = int(
        w.shape[1] / channel
    )  # number of input features (without the channel dimension)

    # assuming channels_first
    if data_format == "channels_first":
        w = K.reshape(w, (-1, channel, n_in) + w.shape[2:])  # (None, c, n_in, n_h)
    else:
        w = K.reshape(w, (-1, n_in, channel) + w.shape[2:])  # (None, n_in, c, n_h)

    w_pos: Tensor = K.relu(w)  # (None, c, n_in, n_h) assuming data_format == channels_first
    w_neg: Tensor = w - w_pos  # (None, c, n_in, n_h)

    # expand dimension of x_min and x_max (None, channel, n_in) assuming channels_first
    axis_channel: int
    if data_format == "channels_first":
        x_min_out: Tensor = K.reshape(
            x_min, [-1, channel, n_in] + [1] * n_h
        )  # x_min_out, x_max_out (None, c, n_in, 1..)
        x_max_out: Tensor = K.reshape(x_max, [-1, channel, n_in] + [1] * n_h)
        x_center_out: Tensor = K.reshape(x_center, [-1, channel, n_in] + [1] * n_h)
        axis_channel = 1
        # get the nominal value (warning clipping) (None, c, n_in, 1..)
    else:
        x_min_out: Tensor = K.reshape(
            x_min, [-1, n_in, channel] + [1] * n_h
        )  # x_min_out, x_max_out (None, n_in, c, 1..)
        x_max_out: Tensor = K.reshape(x_max, [-1, n_in, channel] + [1] * n_h)
        x_center_out: Tensor = K.reshape(x_center, [-1, n_in, channel] + [1] * n_h)
        # get the nominal value (warning clipping) (None, n_in, c, 1..)
        axis_channel = 2

    mask_xai_out: Tensor = K.reshape(
        mask_xai, [-1, n_in] + [1] * n_h
    )  # mask_xai_out, mask_free_out (None, n_in, 1..)
    mask_free_out: Tensor = K.reshape(mask_free, [-1, n_in] + [1] * n_h)

    scoring_samples: Tensor = K.sum(
        w_pos * (x_max_out - x_center_out) + w_neg * (x_min_out - x_center_out), axis_channel
    )  # (None, n_in, 1..)

    # get the scoring samples for each dimension (None, n_in, n_h):

    # xai and free should have really low value so not to be considered
    # scoring samples is always positive so we can set the scoring samples to 0 for xai and free if we don't want to select it
    # xai because the indices are already selected and thus the dimension will be set to the nominal value
    # free because the indices are always freed
    scoring_samples_wo_free: Tensor = (
        scoring_samples * (1 - mask_xai_out) * (1 - mask_free_out)
    )  # (None, n_in, n_h)

    # select cardinality samples from 1..n_in not in xai or free to maximize scoring_samples
    # keras.ops.sort is in ascending order. We sort it along axis n_in and take the last values
    # cardinality does not take into account xai indices and free indices that are set to 0

    # ascending order
    if isinstance(cardinality, int):
        # ascending order
        scoring_rank: Tensor = -keras.ops.sort(
            -scoring_samples_wo_free, axis=1
        )  # (None, n_in, n_h)
        threshold: Tensor = scoring_rank[:, cardinality - 1][:, None]  # (None, 1, n_h..)
    else:
        batch_size: int = len(cardinality)
        threshold: Tensor = -keras.ops.sort(-scoring_samples_wo_free, axis=1)[
            np.arange(batch_size), cardinality - 1
        ][:, None]
        # (None, 1, n_h..)

    # keep only scoring_samples lower than threshold
    final_score_mask: Tensor = K.cast(threshold <= scoring_samples_wo_free, "int")
    final_score: Tensor = K.sum(
        scoring_samples_wo_free * final_score_mask, axis=1
    )  # (None, n_h, ...)

    bias: Tensor = b + K.sum(
        K.sum(w * x_center_out, axis=axis_channel), axis=1
    )  # (None, n_h...) sum over channel
    # update bias with free indices !
    free_scoring_samples: Tensor = K.sum(scoring_samples * mask_free_out, 1)

    bias = bias + free_scoring_samples
    return final_score + bias


def get_lower_box_l0(
    x_min: Tensor,
    x_max: Tensor,
    x_center: Tensor,
    w: Tensor,
    b: Tensor,
    mask_xai: np.ndarray,
    mask_free: np.ndarray,
    channel: int,
    data_format: str,
    cardinality: Union[int, List[int]],
    **kwargs: Any,
) -> Tensor:
    """Computes the lower bound of a linear operation over a hybrid L-infinity/L0 domain.

    This function is the counterpart to `get_upper_box_l0`. It computes the tightest
    possible lower bound for a linear operation (`w*x + b`) over the same complex
    perturbation domain, which combines L-infinity and L0-norm constraints.

    The implementation leverages the duality principle that the minimum of a function
    is the negative of the maximum of its negative, i.e.,
    $min(f(x)) = -max(-f(x))$.
    It computes the lower bound by calling `get_upper_box_l0` with negated
    weights (`-w`) and bias (`-b`) and then negating the result. This effectively
    finds the perturbation that minimizes the linear function's output.

    Args:
        x_min: Tensor defining the lower bounds of the L-infinity component of the domain.
        x_max: Tensor defining the upper bounds of the L-infinity component of the domain.
        x_center: Tensor for the nominal center of the perturbation domain.
        w: Weight tensor of the linear layer.
        b: Bias tensor of the linear layer.
        mask_xai: A binary mask identifying features that are set to their nominal value
        mask_free: A binary mask identifying features that are always perturbed
            under the L-infinity norm.
        channel: The number of channels in the input data.
        data_format: The data format, either "channels_first" or "channels_last".
        cardinality: The L0 norm budget. The maximum number of features from the
            `xai_mask` pool to perturb. Can be an integer for the entire batch or
            a list of integers for per-sample budgets.
        **kwargs: Additional keyword arguments, passed to the underlying
            `get_upper_box_l0` function.

    Returns:
        A tensor representing the computed lower bound of the linear operation.
    """
    return -get_upper_box_l0(
        x_min=x_min,
        x_max=x_max,
        x_center=x_center,
        w=-w,
        b=-b,
        mask_xai=mask_xai,
        mask_free=mask_free,
        channel=channel,
        data_format=data_format,
        cardinality=cardinality,
        **kwargs,
    )


def check_is_robust(
    model, input_sample, eps, channel, data_format, n_class, decomon_model=None
) -> bool:
    """Checks the L-infinity robustness of a model for a given input and epsilon.

    This function verifies whether the model's prediction for a given `input_sample`
    remains constant within an $L_\infty$ ball of radius `eps`. The perturbation
    space is clipped to the valid data range of [0, 1].

    It uses abstract interpretation via the `get_abstract_output_domain` function
    to compute a sound upper bound on the logit differences ($z_{gt} - z_{other}$)
    over the entire input perturbation region, where $z_{gt}$ is the logit of the
    predicted class for the original `input_sample`.

    If the maximum of these upper bounds is less than or equal to zero, it
    formally proves that the original prediction is robust for any input within
    the specified $L_\infty$ ball.

    Args:
        model: The Keras model to verify.
        input_sample: A single input point (e.g., an image) around which
            robustness is checked.
        eps: The radius (epsilon) of the $L_\infty$ norm perturbation.
        channel: The number of channels in the input data.
        data_format: The data format, either "channels_first" or "channels_last".
        n_class: The number of output classes of the model.
        decomon_model: An optional, pre-compiled decomon model for improved
            performance.

    Returns:
        `True` if the model is provably robust for the given input and
        epsilon, `False` otherwise.
    """

    n_in_wo_channel: int = int(input_sample.shape[-1] / channel)
    free_indices: list[int] = [i for i in range(n_in_wo_channel)]

    upper: np.array = get_abstract_output_domain(
        model=model,
        input_sample=input_sample,
        lower_bound=np.maximum(input_sample - eps, 0.0),
        upper_bound=np.minimum(input_sample + eps, 1.0),
        xai_indices=[],
        free_indices=free_indices,
        channel=channel,
        data_format=data_format,
        n_class=n_class,
        decomon_model=decomon_model,
    )  # (1, n_out)

    return np.max(upper) <= 0
