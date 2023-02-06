import torch
from torch import nn, Tensor
from stochman.nnj.abstract_jacobian import AbstractJacobian

from typing import Optional, Tuple, List, Union


class Reshape(AbstractJacobian, nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
        
    def forward(self, x: Tensor) -> Tensor:
        val = x.reshape(x.shape[0], *self.dims)
        return val

    def _jvp(
        self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input"
    ) -> Tensor:
        """
        jacobian vector product
        """
        if wrt == "input":
            return vector
        elif wrt == "weight":
            return None

    def _vjp(
        self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input"
    ) -> Tensor:
        """
        vector jacobian product
        """
        if wrt == "input":
            return vector
        elif wrt == "weight":
            return None

    def _jmp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        jacobian matrix product
        """
        if wrt == "input":
            return matrix
        elif wrt == "weight":
            return None

    def _mjp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        matrix jacobian product
        """
        if wrt == "input":
            return matrix
        elif wrt == "weight":
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
        if matrix is None:
            matrix = torch.ones_like(val)
            from_diag = True
        if wrt == "input":
            if not from_diag and not to_diag:
                # full -> full
                return matrix
            elif from_diag and not to_diag:
                # diag -> full
                return torch.diag_embed(matrix)
            elif not from_diag and to_diag:
                # full -> diag
                return torch.diagonal(matrix, dim1=1, dim2=2)
            elif from_diag and to_diag:
                # diag -> diag
                return matrix
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

    def _jTmjp_batch2(
        self,
        x1: Tensor,
        x2: Tensor,
        val1: Union[Tensor, None],
        val2: Union[Tensor, None],
        matrixes: Union[Tuple[Tensor, Tensor, Tensor], None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tuple[Tensor, Tensor, Tensor], None]:
        """
        jacobian.T matrix jacobian product, computed on 2 inputs (considering cross terms)
        """
        assert x1.shape == x2.shape
        if wrt == "input":
            if not from_diag and not to_diag:
                # full -> full
                return matrixes
            elif from_diag and not to_diag:
                # diag -> full
                return tuple(torch.diag_embed(m) for m in matrixes)
            elif not from_diag and to_diag:
                # full -> diag
                return tuple(torch.diagonal(m, dim1=1, dim2=2) for m in matrixes)
            elif from_diag and to_diag:
                # diag -> diag
                return matrixes
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None
