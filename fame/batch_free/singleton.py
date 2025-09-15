import keras
import numpy as np
from fame.batch_free.abstract import get_abstract_output_domain_singleton


def free_domain_with_abstract_interpretation_singleton(model:keras.models.Model,
                                                       input_sample:np.ndarray,
                                                       lower_bound:np.ndarray,
                                                       upper_bound:np.ndarray,
                                                       xai_indices:list[int],
                                                       free_indices:list[int],
                                                       channel:int=1,
                                                       data_format:str='channels_first',
                                                       n_class:int=10
                                                       )->list[int]:


    # expand one dimension in the set of remaining features to the
    n_in:int = input_sample.shape[-1]
    remaining_indices:list[int] = [i for i in range(n_in) if i not in xai_indices+free_indices]
    upper:np.ndarray
    upper = get_abstract_output_domain_singleton(model=model,
                                                  input_sample=input_sample,
                                                  lower_bound=lower_bound,
                                                  upper_bound=upper_bound,
                                                  xai_indices=xai_indices,
                                                  free_indices=free_indices,
                                                  channel=channel,
                                                  data_format=data_format,
                                                  n_class=n_class)

    if np.min(np.max(upper, -1))<=0:
        return [remaining_indices[np.argmin(np.max(upper, -1))]]
    else:
        return []
