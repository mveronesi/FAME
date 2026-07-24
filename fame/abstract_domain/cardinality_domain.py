from typing import Any, List, Union

import numpy as np
from decomon.perturbation_domain import PerturbationDomain
from fame.abstract_domain.utils import (
    get_lower_ball_l0,
    get_lower_box_l0,
    get_upper_ball_l0,
    get_upper_box_l0,
)
from keras import KerasTensor as Tensor


## Perturbation domain
# prototype that should be adapted to other dimension
class XAIDomain(PerturbationDomain):
    """Defines a hybrid abstract domain for features perturbed under combined L-infinity and L0 norms.

    This class implements a `PerturbationDomain` specifically for eXplainable AI (XAI)
    scenarios. It models a complex input space where:
    1.  A set of "free" features (`free_indices`) can always vary within a
        standard hyper-rectangle (L-infinity perturbation).
    2.  A set of "XAI" features (`xai_indices`) are fixed to their nominal values
    3.  All other features are considered from being candidates for perturbation but only a maximum of `k` (`cardinalities`) can be perturbed at once
        (L0-norm constraint).

    The input tensor `x` representing this domain is expected to have three components:
    the lower bounds, the upper bounds, and the center (nominal) values of the domain. Note that center is not necessarily equal to (lower+upper)/2. as the domain may be clipped

    Attributes:
        n_dim: The number of input dimensions (excluding channels).
        channel: The number of channels in the input.
        data_format: The ordering of dimensions, "channels_first" or "channels_last".
        xai_mask: A binary mask indicating which features are candidates for
            L0-norm perturbation.
        free_mask: A binary mask indicating which features are always perturbed
            under an L-infinity norm.
        cardinalities: The L0-norm budget, i.e., the max number of input features
            that can be perturbed.
    """

    def __init__(
        self,
        xai_indices: List[int],
        free_indices: List[int],
        cardinalities: Union[int, List[int]],
        n_dim: int,
        channel: int,
        data_format: str = "channels_first",
        norm: float = 2,
        eps: Union[float, None] = None,
        *kwargs: Any,
    ):
        """Initializes the XAIDomain.

        Args:
            xai_indices: Indices of features that are fixed to their nominal values
            free_indices: Indices of features that are always perturbed within
                their L-infinity bounds.
            cardinalities: The L0-norm budget. This is the maximum number of
                features from `xai_indices` that can be perturbed simultaneously.
            n_dim: The number of input features (without channels).
            channel: The number of channels for each feature.
            data_format: The data format, either "channels_first" or
                "channels_last".
            *kwargs: Additional arguments passed to the parent class constructor.
        """
        super().__init__(*kwargs)

        if data_format not in ["channels_first", "channels_last"]:
            raise ValueError("unknown data format {}".format(data_format))
        self.n_dim: int = n_dim  # input dimension
        self.channel: int = channel  # channel dimension
        self.data_format: str = data_format  # data_format either channels_first or channels_last

        # mask over xai features and irrelevant features
        self.xai_mask: np.ndarray = np.zeros((self.n_dim,), dtype="float32")
        self.xai_mask[xai_indices] = 1  # 1 if potential candidate to add in the xai features

        self.free_mask: np.ndarray = np.zeros((self.n_dim,), dtype="float32")
        self.free_mask[free_indices] = 1  # 1 if this index has already been free

        self.free_mask = self.free_mask[None]  # (1, n_dim)
        self.xai_mask = self.xai_mask[None]  # (1, n_dim)

        self.cardinalities: Union[
            int, List[int]
        ] = cardinalities  # if list of int, WARNING: the length is equal to batchsize
        self.norm: float = norm
        self.eps: Union[float, None] = eps

        if self.norm not in [np.inf, 2]:
            raise ValueError("unsupported norm {}: only np.inf and 2 are supported".format(self.norm))
        if self.norm == 2 and self.eps is None:
            raise ValueError("eps must be provided when norm=2")

    def get_nb_x_components(self) -> int:
        """Returns the number of components in the input domain tensor `x`.

        For this domain, the input tensor `x` is composed of three parts:
        lower bounds, upper bounds, and center values.

        Returns:
            The integer 3.
        """
        return 3

    def get_upper(self, x: Tensor, w: Tensor, b: Tensor, **kwargs: Any) -> Tensor:
        """Computes the upper bound of a linear operation over the hybrid domain.

        This method implements the abstract transformer for finding the upper
        bound of `w*x + b`, where `x` is an element of this hybrid L-inf/L0 domain.

        Args:
            x: The input tensor representing the domain, with shape
               `(batch, 3, features)`.
            w: The weight tensor of the linear operation.
            b: The bias tensor of the linear operation.
            **kwargs: Additional arguments for the bound computation.

        Returns:
            The tensor of upper bounds.
        """
        x_min: Tensor = self.get_lower_x(x)
        x_max: Tensor = self.get_upper_x(x)
        x_center: Tensor = self.get_center_x(x)

        if self.norm == 2:
            if self.eps is None:
                raise ValueError("eps must be provided when norm=2")
            return get_upper_ball_l0(
                x_center=x_center,
                w=w,
                b=b,
                mask_xai=self.xai_mask,
                mask_free=self.free_mask,
                channel=self.channel,
                data_format=self.data_format,
                cardinality=self.cardinalities,
                eps=float(self.eps),
                **kwargs,
            )

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
        """Computes the lower bound of a linear operation over the hybrid domain.

        This method implements the abstract transformer for finding the lower
        bound of `w*x + b`, where `x` is an element of this hybrid L-inf/L0 domain.

        Args:
            x: The input tensor representing the domain, with shape
               `(batch, 3, features)`.
            w: The weight tensor of the linear operation.
            b: The bias tensor of the linear operation.
            **kwargs: Additional arguments for the bound computation.

        Returns:
            The tensor of lower bounds.
        """
        x_min: Tensor = self.get_lower_x(x)
        x_max: Tensor = self.get_upper_x(x)
        x_center: Tensor = self.get_center_x(x)

        if self.norm == 2:
            if self.eps is None:
                raise ValueError("eps must be provided when norm=2")
            return get_lower_ball_l0(
                x_center=x_center,
                w=w,
                b=b,
                mask_xai=self.xai_mask,
                mask_free=self.free_mask,
                channel=self.channel,
                data_format=self.data_format,
                cardinality=self.cardinalities,
                eps=float(self.eps),
                **kwargs,
            )

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
        """Extracts the upper bound component from the input domain tensor `x`.

        Args:
            x: The input tensor with shape `(batch, 3, features)`.

        Returns:
            The slice corresponding to the upper bounds.
        """
        return x[:, 1]

    def get_lower_x(self, x: Tensor) -> Tensor:
        """Extracts the lower bound component from the input domain tensor `x`.

        Args:
            x: The input tensor with shape `(batch, 3, features)`.

        Returns:
            The slice corresponding to the lower bounds.
        """
        return x[:, 0]

    def get_center_x(self, x: Tensor) -> Tensor:
        """Extracts the center (nominal) value from the input domain tensor `x`.

        Args:
            x: The input tensor with shape `(batch, 3, features)`.

        Returns:
            The slice corresponding to the center values.
        """
        return x[:, 2]
