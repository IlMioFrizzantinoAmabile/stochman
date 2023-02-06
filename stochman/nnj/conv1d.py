import torch
import torch.nn.functional as F
from torch import nn, Tensor
from stochman.nnj.abstract_jacobian import AbstractJacobian

from typing import Optional, Tuple, List, Union

class Conv1d(AbstractJacobian, nn.Conv1d):
    
    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        b, c1, l1 = x.shape
        c2, l2 = val.shape[1:]
        return (
            F.conv1d(
                jac_in.movedim((1, 2), (-2, -1)).reshape(-1, c1, l1),
                weight=self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            .reshape(b, *jac_in.shape[3:], c2, l2)
            .movedim((-2, -1), (1, 2))
        )

    def _jmp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        jacobian matrix product
        """
        if wrt == "input":
            raise NotImplementedError
        elif wrt == "weight":
            raise NotImplementedError