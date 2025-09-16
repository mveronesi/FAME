import keras
import numpy as np
import torch
from fame.batch_free.order import get_greedy_order
from keras import KerasTensor as Tensor

from .fgsm import fast_gradient_method
from .pgd import projected_gradient_descent


def find_singleton_feature_2_add(
    model: keras.models.Model,
    gt_label: int,
    input_sample: Tensor,
    eps: float = 0.0,
    xai_indices: list[int] = [],
    free_indices: list[int] = [],
    method: str = "fgsm",
    device: str = "mps",
    channel: int = 1,
) -> list[int]:
    model.to(device)
    model.eval()

    # create n input domain
    n_in_wo_channel: int = int(input_sample.shape[-1] / channel)
    batch_size = n_in_wo_channel - len(xai_indices) - len(free_indices)

    lower_bound_input_ = np.maximum(input_sample - eps, 0 * input_sample)
    upper_bound_input_ = np.minimum(input_sample + eps, 0 * input_sample + 1)

    # repeat subdomains
    # freeze xai features to the nominal value
    lower_bound_np = np.copy(input_sample) + 0.0
    upper_bound_np = np.copy(input_sample) + 0.0
    if len(free_indices):
        lower_bound_np[free_indices] = lower_bound_input_[free_indices]
        upper_bound_np[free_indices] = upper_bound_input_[free_indices]

    # repeat
    lower_bound_batch = np.repeat(
        lower_bound_np[None], repeats=batch_size, axis=0
    )  # (batch_size, n_in)
    upper_bound_batch = np.repeat(
        upper_bound_np[None], repeats=batch_size, axis=0
    )  # (batch_size, n_in)
    input_sample_batch = np.repeat(
        input_sample[None], repeats=batch_size, axis=0
    )  # (batch_size, n_in)

    # expand one dimension in the set of remaining features to the
    remaining_indices = [i for i in range(n_in_wo_channel) if i not in xai_indices + free_indices]
    for i, j in enumerate(remaining_indices):
        lower_bound_batch[i, j] = lower_bound_input_[j]
        upper_bound_batch[i, j] = upper_bound_input_[j]

    # ----- tensors on the same device
    lower_t = torch.tensor(
        lower_bound_batch, dtype=torch.float32, device=device
    )  # (batch_size,n_in)
    upper_t = torch.tensor(
        upper_bound_batch, dtype=torch.float32, device=device
    )  # (batch_size,n_in)
    input_t = torch.tensor(
        input_sample_batch, dtype=torch.float32, device=device
    )  # (batch_size, n_in)

    gt_label_t = torch.tensor(np.array([gt_label] * batch_size, dtype="int64")).to(device)

    if method == "fgsm":
        x_adv_class = fast_gradient_method(
            model_fn=model,
            x=input_t,
            eps=eps,
            loss_fn=torch.nn.CrossEntropyLoss(),
            norm=np.inf,
            clip_min=lower_t,
            clip_max=upper_t,
            y=gt_label_t,
        )
    else:
        x_adv_class = projected_gradient_descent(
            model_fn=model,
            x=input_t,
            eps=eps,
            loss_fn=torch.nn.CrossEntropyLoss(),
            norm=np.inf,
            clip_min=lower_t,
            clip_max=upper_t,
            y=gt_label_t,
        )
    adv_pred = model.predict(x_adv_class, verbose=0).argmax(-1)

    xai_set: list[int] = [
        remaining_indices[j] for j in range(batch_size) if adv_pred[j] != gt_label
    ]
    return xai_set


def find_closest_xai(
    model: keras.models.Model,
    gt_label: int,
    input_sample: Tensor,
    eps: float = 0.0,
    xai_indices: list[int] = [],
    free_indices: list[int] = [],
    method: str = "fgsm",
    device: str = "mps",
    channel: int = 1,
    data_format: int = "channels_first",
    n_class: int = 10,
    traversal_order: str = "greedy",
) -> tuple[list[int], list[int]]:
    n_in_wo_channel: int = int(input_sample.shape[-1] / channel)
    n_remaining = n_in_wo_channel - len(xai_indices) - len(free_indices)

    lower_bound: np.ndarray = np.maximum(input_sample - eps, 0 * input_sample)
    upper_bound: np.ndarray = np.minimum(input_sample + eps, 0 * input_sample + 1)

    if traversal_order == "greedy":
        remaining_features_with_traversal = get_greedy_order(
            model=model,
            input_sample=input_sample,
            gt_label=gt_label,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            xai_indices=xai_indices,
            free_indices=free_indices,
            channel=channel,
            data_format=data_format,
            n_class=n_class,
        )
    else:
        raise NotImplementedError("implement other orders if needed")

    potential_xai: list[int] = []
    for i in range(n_in_wo_channel):
        potential_xai = find_singleton_feature_2_add(
            model=model,
            gt_label=gt_label,
            input_sample=input_sample,
            eps=eps,
            xai_indices=xai_indices,
            free_indices=free_indices + remaining_features_with_traversal[:i],
            method=method,
            device=device,
            channel=channel,
        )
        if i + len(potential_xai) >= n_remaining - 2:
            break

    return potential_xai, remaining_features_with_traversal[:i]
