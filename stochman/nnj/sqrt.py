import torch
from torch import nn, Tensor
from stochman.nnj.abstract_jacobian import AbstractDiagonalJacobian

from typing import Optional, Tuple, List, Union

class Sqrt(AbstractDiagonalJacobian, nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        val = torch.sqrt(x)
        return val

    def _jacobian(
        self, x: Tensor, val: Union[Tensor, None] = None, wrt: str = "input", diag: bool = True
    ) -> Union[Tensor, None]:
        """Returns the Jacobian matrix"""
        if wrt == "input":
            if val is None:
                val = self.forward(x)
            diag_jacobian = 0.5 / val
            if diag:
                return diag_jacobian
            else:
                return torch.diag_embed(diag_jacobian)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None