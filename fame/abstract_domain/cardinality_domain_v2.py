# cardinality domain with extra inputs: one hot encoding for xai_indices and free_indices

from typing import Any, List, Union

import numpy as np
from decomon.perturbation_domain import PerturbationDomain, get_upper_box
from fame.abstract_domain.utils import get_lower_box_l0, get_upper_box_l0
from keras import KerasTensor as Tensor


## Perturbation domain
# prototype that should be adapted to other dimension
class XAIDomainWithTraversal(PerturbationDomain):
    def __init__(
        self,
        n_dim: int,
        channel: int,
        data_format: str = "channels_first",
        *kwargs: Any,
    ):
        super().__init__(*kwargs)

        if data_format not in ["channels_first", "channels_last"]:
            raise ValueError("unknown data format {}".format(data_format))
        self.n_dim: int = n_dim  # input dimension
        self.channel: int = channel  # channel dimension
        self.data_format: str = data_format  # data_format either channels_first or channels_last

    def get_nb_x_components(self) -> int:
        return 5 # (lower, upper, center, xai_indices, free_indices)

    def get_upper(self, x: Tensor, w: Tensor, b: Tensor, **kwargs: Any) -> Tensor:
        x_min: Tensor = self.get_lower_x(x)
        x_max: Tensor = self.get_upper_x(x)
        x_center: Tensor = self.get_center_x(x)

        res: Tensor = get_upper_box_l0(
            x_min=x_min,
            x_max=x_max,
            x_center=x_center,
            w=w,
            b=b,
            mask_xai=self.xai_mask,
            mask_free=self.free_mask,
            channel=self.channel,
            data_format=self.data_format,
            cardinality=self.cardinalities,
            **kwargs,
        )
        return res

    def get_lower(self, x: Tensor, w: Tensor, b: Tensor, **kwargs: Any) -> Tensor:
        x_min: Tensor = self.get_lower_x(x)
        x_max: Tensor = self.get_upper_x(x)
        x_center: Tensor = self.get_center_x(x)

        return get_lower_box_l0(
            x_min=x_min,
            x_max=x_max,
            x_center=x_center,
            w=w,
            b=b,
            mask_xai=self.xai_mask,
            mask_free=self.free_mask,
            channel=self.channel,
            data_format=self.data_format,
            cardinality=self.cardinalities,
            **kwargs,
        )

    def get_upper_x(self, x: Tensor) -> Tensor:
        return x[:, 1]

    def get_lower_x(self, x: Tensor) -> Tensor:
        return x[:, 0]

    def get_center_x(self, x: Tensor) -> Tensor:
        return x[:, 2]
    
    def get_xai_indices(self, x:Tensor)->Tensor:
