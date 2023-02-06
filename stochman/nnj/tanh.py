import torch
from torch import nn, Tensor
from stochman.nnj.abstract_jacobian import AbstractDiagonalJacobian

from typing import Optional, Tuple, List, Union

class Tanh(AbstractDiagonalJacobian, nn.Tanh):

    def _jacobian(
        self, x: Tensor, val: Union[Tensor, None] = None, wrt: str = "input", diag: bool = True
    ) -> Union[Tensor, None]:
        """Returns the Jacobian matrix"""
        if wrt == "input":
            if val is None:
                val = self.forward(x)
            diag_jacobian = (torch.ones(val.shape, device=val.device) - val**2).reshape(val.shape[0], -1)
            if diag:
                return diag_jacobian
            else:
                return torch.diag_embed(diag_jacobian)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None
