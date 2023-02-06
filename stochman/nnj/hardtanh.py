import torch
from torch import nn, Tensor
from stochman.nnj.abstract_jacobian import AbstractDiagonalJacobian

from typing import Optional, Tuple, List, Union

class Hardtanh(AbstractDiagonalJacobian, nn.Hardtanh):

    def _jacobian(
        self, x: Tensor, val: Union[Tensor, None] = None, wrt: str = "input", diag: bool = True
    ) -> Union[Tensor, None]:
        """Returns the Jacobian matrix"""
        if wrt == "input":
            if val is None:
                val = self.forward(x)
            diag_jacobian = torch.zeros_like(val)
            diag_jacobian[val.abs() < 1.0] = 1.0
            if diag:
                return diag_jacobian
            else:
                return torch.diag_embed(diag_jacobian)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None
