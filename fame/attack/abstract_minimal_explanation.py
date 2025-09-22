import keras
import numpy as np
import torch
from fame.batch_free.order import get_greedy_order
from keras import KerasTensor as Tensor


from .utils import get_attacks_bounds
from .attack import attack, find_singleton_feature_2_add

from .abstract_minimal_explanation_sequential import find_closest_xai as find_closest_xai_singleton



def find_closest_xai_with_dichotomy(
    model: keras.models.Model,
    gt_label: int,
    input_sample: Tensor,
    eps: float = 0.0,
    xai_indices: list[int] = [],
    free_indices: list[int] = [],
    #remaining_indices:list[int] =[],
    method: str = "fgsm",
    device: str = "mps",
    channel: int = 1,
    data_format: int = "channels_first",
    n_class: int = 10,
    traversal_order: str = "greedy",
) -> tuple[list[int], list[int]]:
    
   
    # compute attacks on every remaining feature
    n_in_wo_channel:int = int(input_sample.shape[-1]/channel)
    remaining_indices = [i for i in range(n_in_wo_channel) if not i in xai_indices+free_indices]

    if len(remaining_indices)==0:
        # best solution is
        assert len(xai_indices+free_indices)==n_in_wo_channel, "missing input features"
        return xai_indices, free_indices

    lower_bound: np.ndarray = np.maximum(input_sample - eps, 0 * input_sample)
    upper_bound: np.ndarray = np.minimum(input_sample + eps, 0 * input_sample + 1)


    # start by attacking everything except xai_indices
    input_sample_everything, lower_bound_everything,  upper_bound_everything= get_attacks_bounds(input_sample=input_sample,
                       eps=eps,
                       free_indices=free_indices+remaining_indices[1:],
                       remaining_indices=remaining_indices[:1],
                       channel=channel,
                       data_format=data_format)
    adv_pred_everything:np.array = attack(model=model, 
           input_sample_batch=input_sample_everything, 
           lower_bound_batch=lower_bound_everything, 
           upper_bound_batch=upper_bound_everything,
           gt_label=gt_label,
           eps=eps, 
           method=method,
           ) #(1,)
    if adv_pred_everything[0]==gt_label:
        return xai_indices, free_indices+remaining_indices
    
    xai_set_init = find_singleton_feature_2_add(model=model,
                                                gt_label=gt_label,
                                                input_sample=input_sample,
                                                eps=eps,
                                                free_indices=free_indices,
                                                remaining_indices=remaining_indices,
                                                method=method,
                                                device=device,
                                                channel=channel,
                                                data_format=data_format
                                                )    
    
    # remove them from remaining_indices
    remaining_indices_init:list[int] = [i for i in remaining_indices if not i in xai_set_init]
    # stop criterion: there remains at most one index
    if len(remaining_indices_init)<=1:
        # best solution is
        assert len(xai_set_init+xai_indices+remaining_indices_init+free_indices), "missing input features D"
        return xai_set_init+xai_indices, remaining_indices_init+free_indices

    # compute traversal order
    if traversal_order == "greedy":
        remaining_features_with_traversal = get_greedy_order(
            model=model,
            input_sample=input_sample,
            gt_label=gt_label,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            xai_indices=xai_indices+xai_set_init,
            free_indices=free_indices,
            channel=channel,
            data_format=data_format,
            n_class=n_class,
        )
    else:
        raise NotImplementedError("implement other orders if needed")
    
    assert len(remaining_features_with_traversal)==len(remaining_indices_init), 'missing remaining indices'
    # split in half
    n:int = int(len(remaining_features_with_traversal)/2)
    remaining_indices_part_0:list[int] = remaining_features_with_traversal[:n]
    remaining_indices_part_1:list[int] = remaining_features_with_traversal[n:]

    # attack on the second half
    # solution A
    xai_A = find_singleton_feature_2_add(model=model,
                                                    gt_label=gt_label,
                                                    input_sample=input_sample,
                                                    eps=eps,
                                                    free_indices=free_indices+remaining_indices_part_0,
                                                    remaining_indices=remaining_indices_part_1,
                                                    method=method,
                                                    device=device,
                                                    channel=channel,
                                                    data_format=data_format
                                                    )

    # if len(xai_A)==0, then freeing  remaining_indices_part_0 is not enough
    if len(xai_A)==0:

        xai_C, remaining_C = find_closest_xai_with_dichotomy(model=model,
                                                         gt_label=gt_label,
                                                        input_sample=input_sample,
                                                        eps=eps,
                                                        xai_indices=xai_indices+xai_set_init,
                                                        free_indices=free_indices+remaining_indices_part_0,
                                                        #remaining_indices=remaining_indices_part_0+xai_A,
                                                        method=method,
                                                        device=device,
                                                        channel=channel,
                                                        data_format=data_format,
                                                        n_class=n_class,
                                                        traversal_order=traversal_order
                                                        )
        
        assert len(xai_C+remaining_C)==n_in_wo_channel, "missing input features C"
        return xai_C, remaining_C
    
    # since we reduce the perturbation domain, free_indices_0_1 will not be attackable
    # hence we add them in the free set
    free_indices_A:list[int] = [i for i in remaining_indices_part_1 if not i in xai_A]
    # we assume at first that we can add in the explanation xai_A

    xai_B:list[int]
    remaining_B:list[int]


    xai_B, remaining_B = find_closest_xai_with_dichotomy(model=model,
                                                         gt_label=gt_label,
                                                        input_sample=input_sample,
                                                        eps=eps,
                                                        xai_indices=xai_indices+xai_set_init+xai_A,
                                                        free_indices=free_indices+free_indices_A,
                                                        method=method,
                                                        device=device,
                                                        channel=channel,
                                                        data_format=data_format,
                                                        n_class=n_class,
                                                        traversal_order=traversal_order
                                                        )
    # filter xai_A from xai_B
    xai_B = [i for i in xai_B if not i in xai_A]
    
    # we attack again xai_A features 
    xai_B_A:list[int] = find_singleton_feature_2_add(model=model,
                                                gt_label=gt_label,
                                                input_sample=input_sample,
                                                eps=eps,
                                                free_indices=remaining_B,
                                                remaining_indices=xai_A,
                                                method=method,
                                                device=device,
                                                channel=channel,
                                                data_format=data_format
                                                ) 
    remaining_B +=[i for i in xai_A if not i in xai_B_A]

    # call on part 2

    # solution A: candidate = free_indices_A + remaining_indices_part_0 + free_indices, 
    #             xai = xai_set_init + xai_A + xai_indices
    # solution B: candidate = remaining_B, xai = xai_B + xai_B_A

    # best solution is the one that minimize the distance to minimal abductive explanation
    if len(free_indices_A+remaining_indices_part_0+free_indices)< len(remaining_B):
        assert len(xai_set_init + xai_A + xai_indices+remaining_indices_part_0 + free_indices_A + free_indices)==n_in_wo_channel, 'missing input features A'
        return xai_set_init + xai_A + xai_indices, remaining_indices_part_0 + free_indices_A + free_indices
    else:
        assert len(xai_B+xai_B_A+remaining_B)==n_in_wo_channel, "missing input features B"
        return xai_B + xai_B_A, remaining_B

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
    
    n_in_wo_channel:int = int(input_sample.shape[-1]/channel)
    
    potential_xai_d, _ =  find_closest_xai_with_dichotomy(model=model,
                                           gt_label=gt_label,
                                           input_sample=input_sample,
                                           eps=eps,
                                           xai_indices=xai_indices,
                                           free_indices=free_indices,
                                           method=method,
                                           device=device,
                                           channel=channel,
                                           data_format=data_format,
                                           n_class=n_class,
                                           traversal_order=traversal_order
                                           )

    # relaunch with sequential (longer but tighter so we do it on a restricted domain)
    if n_in_wo_channel > len(potential_xai_d+free_indices):
        potential_xai_s, extra_free_s = find_closest_xai_singleton(model=model,
                                            gt_label=gt_label,
                                            input_sample=input_sample,
                                            eps=eps,
                                            xai_indices=potential_xai_d,
                                            free_indices=free_indices,
                                            method=method,
                                            device=device,
                                            channel=channel,
                                            data_format=data_format,
                                            n_class=n_class,
                                            traversal_order=traversal_order
                                            )
        # check
        assert len(potential_xai_d + potential_xai_s+extra_free_s+free_indices)==n_in_wo_channel, "missing input features"
        return potential_xai_d + potential_xai_s, extra_free_s
    else:
        # we have already a minimal explanation
        return potential_xai_d, []