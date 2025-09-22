import keras
import torch
import numpy as np
from keras import KerasTensor as Tensor
from .fgsm import fast_gradient_method
from .pgd import projected_gradient_descent
from .utils import get_attacks_bounds


def attack(model:keras.models.Model, 
           input_sample_batch:np.ndarray, 
           lower_bound_batch:np.ndarray, 
           upper_bound_batch:np.ndarray,
           gt_label:int,
           eps:float, 
           method:str,
           device:str="mps")->np.ndarray:
    

    model.to(device)
    input_t:Tensor=torch.tensor(input_sample_batch, dtype=torch.float32, device=device)
    lower_t:Tensor=torch.tensor(lower_bound_batch, dtype=torch.float32, device=device)
    upper_t:Tensor=torch.tensor(upper_bound_batch, dtype=torch.float32, device=device)
    batch_size = len(input_sample_batch)
    gt_label_t:Tensor=torch.tensor(np.array([gt_label] * batch_size, dtype="int64"), dtype=torch.int64, device=device)

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
        eps_iter = eps/10. # arbitraty (do a config file)
        nb_iter = 50 # arbitraty (do a config file)
        x_adv_class = projected_gradient_descent(
            model_fn=model,
            x=input_t,
            eps=eps,
            eps_iter=eps_iter,
            nb_iter=nb_iter,
            loss_fn=torch.nn.CrossEntropyLoss(),
            norm=np.inf,
            clip_min=lower_t,
            clip_max=upper_t,
            y=gt_label_t,
            rand_init=False, # for reproducibility
        )
    adv_pred = model.predict(x_adv_class, verbose=0).argmax(-1)

    return adv_pred #(batch_size,)

def find_singleton_feature_2_add(
    model: keras.models.Model,
    gt_label: int,
    input_sample: Tensor,
    eps: float = 0.0,
    free_indices: list[int] = [],
    remaining_indices:list[int]=[], # attack only those features
    method: str = "fgsm",
    device: str = "mps",
    channel: int = 1,
    data_format="channels_first"
) -> list[int]:
    model.to(device)
    model.eval()

    input_sample_batch, lower_bound_batch,  upper_bound_batch= get_attacks_bounds(input_sample=input_sample,
                       eps=eps,
                       free_indices=free_indices,
                       remaining_indices=remaining_indices,
                       channel=channel,
                       data_format=data_format)
    batch_size: int = len(remaining_indices)

    adv_pred:np.ndarray=attack(model=model, 
           input_sample_batch=input_sample_batch, 
           lower_bound_batch=lower_bound_batch, 
           upper_bound_batch=upper_bound_batch,
           gt_label=gt_label,
           eps=eps, 
           method=method,
           device=device) # (batch_size,)

    xai_set: list[int] = [
        remaining_indices[j] for j in range(batch_size) if adv_pred[j] != gt_label
    ]
    return xai_set
