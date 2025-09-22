# source: https://github.com/cleverhans-lab/cleverhans/blob/master/cleverhans/torch/utils.py
import numpy as np
import torch




def clip_eta(eta, norm, eps):
    """
    PyTorch implementation of the clip_eta in utils_tf.

    :param eta: Tensor
    :param norm: np.inf, 1, or 2
    :param eps: float
    """
    if norm not in [np.inf, 1, 2]:
        raise ValueError("norm must be np.inf, 1, or 2.")

    avoid_zero_div = torch.tensor(1e-12, dtype=eta.dtype, device=eta.device)
    reduc_ind = list(range(1, len(eta.size())))
    if norm == np.inf:
        eta = torch.clamp(eta, -eps, eps)
    else:
        if norm == 1:
            raise NotImplementedError("L1 clip is not implemented.")
            norm = torch.max(avoid_zero_div, torch.sum(torch.abs(eta), dim=reduc_ind, keepdim=True))
        elif norm == 2:
            norm = torch.sqrt(
                torch.max(avoid_zero_div, torch.sum(eta**2, dim=reduc_ind, keepdim=True))
            )
        factor = torch.min(torch.tensor(1.0, dtype=eta.dtype, device=eta.device), eps / norm)
        eta *= factor
    return eta


def optimize_linear(grad, eps, norm=np.inf):
    """
    Solves for the optimal input to a linear function under a norm constraint.

    Optimal_perturbation = argmax_{eta, ||eta||_{norm} < eps} dot(eta, grad)

    :param grad: Tensor, shape (N, d_1, ...). Batch of gradients
    :param eps: float. Scalar specifying size of constraint region
    :param norm: np.inf, 1, or 2. Order of norm constraint.
    :returns: Tensor, shape (N, d_1, ...). Optimal perturbation
    """

    red_ind = list(range(1, len(grad.size())))
    avoid_zero_div = torch.tensor(1e-12, dtype=grad.dtype, device=grad.device)
    if norm == np.inf:
        # Take sign of gradient
        optimal_perturbation = torch.sign(grad)
    elif norm == 1:
        abs_grad = torch.abs(grad)
        sign = torch.sign(grad)
        red_ind = list(range(1, len(grad.size())))
        abs_grad = torch.abs(grad)
        ori_shape = [1] * len(grad.size())
        ori_shape[0] = grad.size(0)

        max_abs_grad, _ = torch.max(abs_grad.view(grad.size(0), -1), 1)
        max_mask = abs_grad.eq(max_abs_grad.view(ori_shape)).to(torch.float)
        num_ties = max_mask
        for red_scalar in red_ind:
            num_ties = torch.sum(num_ties, red_scalar, keepdim=True)
        optimal_perturbation = sign * max_mask / num_ties
        # TODO integrate below to a test file
        # check that the optimal perturbations have been correctly computed
        opt_pert_norm = optimal_perturbation.abs().sum(dim=red_ind)
        assert torch.all(opt_pert_norm == torch.ones_like(opt_pert_norm))
    elif norm == 2:
        square = torch.max(avoid_zero_div, torch.sum(grad**2, red_ind, keepdim=True))
        optimal_perturbation = grad / torch.sqrt(square)
        # TODO integrate below to a test file
        # check that the optimal perturbations have been correctly computed
        opt_pert_norm = optimal_perturbation.pow(2).sum(dim=red_ind, keepdim=True).sqrt()
        one_mask = (square <= avoid_zero_div).to(torch.float) * opt_pert_norm + (
            square > avoid_zero_div
        ).to(torch.float)
        assert torch.allclose(opt_pert_norm, one_mask, rtol=1e-05, atol=1e-08)
    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are " "currently implemented.")

    # Scale perturbation to be the solution for the norm=eps rather than
    # norm=1 problem
    scaled_perturbation = eps * optimal_perturbation
    return scaled_perturbation


def get_attacks_bounds(input_sample:np.array,
                       eps:float,
                       free_indices:list[int],
                       remaining_indices:list[int],
                       channel:int=1,
                       data_format:str="channels_first")->tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    n_in_with_channel:int = input_sample.shape[-1]
    n_in_wo_channel: int = int(input_sample.shape[-1] / channel)
    batch_size = len(remaining_indices)

    lower_bound_input_ = np.maximum(input_sample - eps, 0 * input_sample)
    upper_bound_input_ = np.minimum(input_sample + eps, 0 * input_sample + 1)

    # reshape according to data_format
    lower_bound_c:np.array
    upper_bound_c:np.array
    input_sample_c:np.array
    if data_format=="channels_first":
        lower_bound_c = np.reshape(lower_bound_input_, (channel, n_in_wo_channel))
        upper_bound_c = np.reshape(upper_bound_input_, (channel, n_in_wo_channel))
        input_sample_c = np.reshape(input_sample, (channel, n_in_wo_channel))
    else:
        lower_bound_c = np.reshape(lower_bound_input_, (n_in_wo_channel, channel))
        upper_bound_c = np.reshape(upper_bound_input_, (n_in_wo_channel, channel))
        input_sample_c = np.reshape(input_sample, (n_in_wo_channel, channel))

    # repeat subdomains
    # freeze xai features to the nominal value
    lower_bound_np = np.copy(input_sample_c) + 0.0 # (channel, n_in_wo_channel) if channels_first
    upper_bound_np = np.copy(input_sample_c) + 0.0

    if len(free_indices):
        if data_format=="channels_first":
            lower_bound_np[:, free_indices] = lower_bound_c[:,free_indices]
            upper_bound_np[:, free_indices] = upper_bound_c[:,free_indices]
        else:
            lower_bound_np[free_indices, :] = lower_bound_c[free_indices, :]
            upper_bound_np[free_indices, :] = upper_bound_c[free_indices, :]
    # repeat
    lower_bound_batch = np.repeat(
        lower_bound_np[None], repeats=batch_size, axis=0
    )  # (batch_size, channel, n_in_wo_channel)
    upper_bound_batch = np.repeat(
        upper_bound_np[None], repeats=batch_size, axis=0
    )  # (batch_size, channel, n_in_wo_channel)
    input_sample_batch = np.repeat(
        input_sample_c[None], repeats=batch_size, axis=0
    )  # (batch_size, channel, n_in_wo_channel)

    # expand one dimension in the set of remaining features
    for i, j in enumerate(remaining_indices):
        if data_format=="channels_first":
            lower_bound_batch[i, :, j] = lower_bound_c[:,j]
            upper_bound_batch[i, :, j] = upper_bound_c[:, j]
        else:
            lower_bound_batch[i, j, :] = lower_bound_c[j,:]
            upper_bound_batch[i, j, :] = upper_bound_c[j,:]

    # reshape to fit the input shape of the model
    lower_bound_batch = np.reshape(lower_bound_batch, (batch_size, n_in_with_channel))
    upper_bound_batch = np.reshape(upper_bound_batch, (batch_size, n_in_with_channel))
    input_sample_batch = np.reshape(input_sample_batch, (batch_size, n_in_with_channel))

    return input_sample_batch, lower_bound_batch, upper_bound_batch


