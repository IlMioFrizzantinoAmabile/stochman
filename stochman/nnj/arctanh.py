import torch
from torch import nn, Tensor
from stochman.nnj.abstract_jacobian import AbstractDiagonalJacobian

from typing import Optional, Tuple, List, Union

class ArcTanh(AbstractDiagonalJacobian, nn.Tanh):
    def forward(self, x: Tensor) -> Tensor:
        xc = x.clamp(
            -(1 - 1e-4), 1 - 1e-4
        )  # the inverse is only defined on [-1, 1] so we project onto this interval
        val = (
            0.5 * (1.0 + xc).log() - 0.5 * (1.0 - xc).log()
        )  # XXX: is it stable to compute log((1+xc)/(1-xc)) ? (that would be faster)
        return val

    def _jacobian(
        self, x: Tensor, val: Union[Tensor, None] = None, wrt: str = "input", diag: bool = True
    ) -> Union[Tensor, None]:
        """Returns the Jacobian matrix"""
        if wrt == "input":
            if val is None:
                val = self.forward(x)
            diag_jacobian = -1.0 / (x**2 - 1.0)
            if diag:
                return diag_jacobian
            else:
                return torch.diag_embed(diag_jacobian)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None