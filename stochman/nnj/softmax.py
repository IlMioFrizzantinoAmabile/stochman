import torch
from torch import nn, Tensor
from stochman.nnj.abstract_jacobian import AbstractJacobian

from typing import Optional, Tuple, List, Union

class Softmax(AbstractJacobian, nn.Softmax):
    
    def _jacobian(
        self, x: Tensor, val: Union[Tensor, None] = None, wrt: str = "input", diag: bool = False
    ) -> Union[Tensor, None]:
        """Returns the Jacobian matrix"""
        if wrt == "input":
            if diag==True:
                raise NotImplementedError
            if self.dim == 0:
                raise ValueError("Jacobian computation not supported for `dim=0`")
            if val is None:
                val = self.forward(x)
            return torch.diag_embed(val) - torch.matmul(val.unsqueeze(-1), val.unsqueeze(-2))
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

    def _jmp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Union[Tensor, None]:
        """
        jacobian matrix product
        """
        if wrt == "input":
            jacobian = self._jacobian(x, val)
            if matrix is None:
                return jacobian
            n = matrix.ndim - jacobian.ndim
            jacobian = jacobian.reshape((1,) * n + jacobian.shape)
            if matrix.ndim == 4:
                return (jacobian @ matrix.permute(3, 0, 1, 2)).permute(1, 2, 3, 0)
            if matrix.ndim == 5:
                return (jacobian @ matrix.permute(3, 4, 0, 1, 2)).permute(2, 3, 4, 0, 1)
            if matrix.ndim == 6:
                return (jacobian @ matrix.permute(3, 4, 5, 0, 1, 2)).permute(3, 4, 5, 0, 1, 2)
            return jacobian @ matrix
        elif wrt == "weight":
            return None