import torch
from torch import nn, Tensor
from stochman.nnj.abstract_jacobian import AbstractDiagonalJacobian

from typing import Optional, Tuple, List, Union

class ReLU(AbstractDiagonalJacobian, nn.ReLU):
    
    def _jacobian(
        self, x: Tensor, val: Union[Tensor, None] = None, wrt: str = "input", diag: bool = True
    ) -> Union[Tensor, None]:
        """Returns the Jacobian matrix"""
        if wrt == "input":
            if val is None:
                val = self.forward(x)
            diag_jacobian = (val > 0.0).type(val.dtype)
            if diag:
                return diag_jacobian
            else:
                return torch.diag_embed(diag_jacobian)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

    def _jTmjp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, None]:
        """
        jacobian.T matrix jacobian product
        """
        b = x.shape[0]
        if val is None:
            val = self.forward(x)
        if matrix is None:
            matrix = torch.ones_like(val).reshape(b, -1)
            from_diag = True
        if wrt == "input":
            if not from_diag and not to_diag:
                # full -> full
                diag_jacobian = self._jacobian(x, val, diag=True)
                return torch.einsum("bi,bik,bk->bik", diag_jacobian, matrix, diag_jacobian)
            elif from_diag and not to_diag:
                # diag -> full
                diag_jacobian_square = self._jacobian(x, val, diag=True)
                return torch.diag_embed(torch.einsum("bi,bi->bi", diag_jacobian_square, matrix))
            elif not from_diag and to_diag:
                # full -> diag
                diag_jacobian = self._jacobian(x, val, diag=True)
                return torch.einsum("bi,bii,bi->bi", diag_jacobian, matrix, diag_jacobian)
            elif from_diag and to_diag:
                # diag -> diag
                diag_jacobian_square = self._jacobian(x, val, diag=True)
                return torch.einsum("bi,bi->bi", diag_jacobian_square, matrix)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None
