import torch
import torch.nn.functional as F
from torch import nn, Tensor
from stochman.nnj.abstract_jacobian import AbstractJacobian

from typing import Optional, Tuple, List, Union

class PosLinear(AbstractJacobian, nn.Linear):
    def forward(self, x: Tensor):
        bias = F.softplus(self.bias) if self.bias is not None else self.bias
        val = F.linear(x, F.softplus(self.weight), bias)
        return val

    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        return F.linear(jac_in.movedim(1, -1), F.softplus(self.weight), bias=None).movedim(-1, 1)

    def _jmp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        jacobian matrix product
        """
        if wrt == "input":
            if val is None:
                val = self.forward(x)
            return self._jacobian_wrt_input_mult_left_vec(x, val, matrix)
        elif wrt == "weight":
            raise NotImplementedError
