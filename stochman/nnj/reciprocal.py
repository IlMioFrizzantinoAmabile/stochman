import torch
from torch import nn, Tensor
from stochman.nnj.abstract_jacobian import AbstractDiagonalJacobian

from typing import Optional, Tuple, List, Union

class Reciprocal(AbstractDiagonalJacobian, nn.Module):
    def __init__(self, b: float = 0.0):
        super().__init__()
        self.b = b

    def forward(self, x: Tensor) -> Tensor:
        val = 1.0 / (x + self.b)
        return val

    def _jacobian(
        self, x: Tensor, val: Union[Tensor, None] = None, wrt: str = "input", diag: bool = True
    ) -> Union[Tensor, None]:
        """Returns the Jacobian matrix"""
        if wrt == "input":
            if val is None:
                val = self.forward(x)
            diag_jacobian = -((val) ** 2)
            if diag:
                return diag_jacobian
            else:
                return torch.diag_embed(diag_jacobian)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None