import torch
import torch.nn.functional as F
from torch import nn, Tensor
from stochman.nnj.abstract_jacobian import AbstractJacobian

from typing import Optional, Tuple, List, Union

class Conv3d(AbstractJacobian, nn.Conv3d):
    
    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        b, c1, d1, h1, w1 = x.shape
        c2, d2, h2, w2 = val.shape[1:]
        return (
            F.conv3d(
                jac_in.movedim((1, 2, 3, 4), (-4, -3, -2, -1)).reshape(-1, c1, d1, h1, w1),
                weight=self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            .reshape(b, *jac_in.shape[5:], c2, d2, h2, w2)
            .movedim((-4, -3, -2, -1), (1, 2, 3, 4))
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