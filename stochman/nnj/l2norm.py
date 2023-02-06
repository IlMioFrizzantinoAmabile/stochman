import torch
from torch import nn, Tensor
from stochman.nnj.abstract_jacobian import AbstractJacobian

from typing import Optional, Tuple, List, Union

class L2Norm(AbstractJacobian, nn.Module):
    """L2 normalization layer"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, eps: float = 1e-6) -> Tensor:
        return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)

    def _jacobian(
        self, x: Tensor, val: Union[Tensor, None] = None, wrt: str = "input", diag: bool = False
    ) -> Union[Tensor, None]:
        """Returns the Jacobian matrix"""
        if wrt == "input":
            if diag==True:
                raise NotImplementedError
            if val is None:
                val = self.forward(x)
            b, d = x.shape
            norm = torch.norm(x, p=2, dim=1)
            normalized_x = torch.einsum("b,bi->bi", 1 / (norm + 1e-6), x)
            jacobian = torch.einsum("bi,bj->bij", normalized_x, normalized_x)
            jacobian = torch.diag(torch.ones(d, device=x.device)).expand(b, d, d) - jacobian
            jacobian = torch.einsum("b,bij->bij", 1 / (norm + 1e-6), jacobian)
            return jacobian
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

    def _jvp(
        self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input"
    ) -> Union[Tensor, None]:
        """
        jacobian vector product
        """
        if wrt == "input":
            jacobian = self._jacobian(x, val)
            return torch.einsum("bij,bj->bi", jacobian, vector)
        elif wrt == "weight":
            return None

    def _vjp(
        self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input"
    ) -> Union[Tensor, None]:
        """
        vector jacobian product
        """
        if wrt == "input":
            jacobian = self._jacobian(x, val)
            return torch.einsum("bi,bij->bj", vector, jacobian)
        elif wrt == "weight":
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
            return torch.einsum("bij,bjk->bik", jacobian, matrix)
        elif wrt == "weight":
            return None

    def _mjp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Union[Tensor, None]:
        """
        matrix jacobian product
        """
        if wrt == "input":
            jacobian = self._jacobian(x, val)
            if matrix is None:
                return jacobian
            return torch.einsum("bij,bjk->bik", matrix, jacobian)
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
        if val is None:
            val = self.forward(x)
        if matrix is None:
            matrix = torch.ones_like(val)
            from_diag = True
        if wrt == "input":
            if not from_diag and not to_diag:
                # full -> full
                jacobian = self._jacobian(x, val)
                return torch.einsum("bij,bik,bkl->bjl", jacobian, matrix, jacobian)
            elif from_diag and not to_diag:
                # diag -> full
                jacobian = self._jacobian(x, val)
                return torch.einsum("bij,bi,bil->bjl", jacobian, matrix, jacobian)
            elif not from_diag and to_diag:
                # full -> diag
                jacobian = self._jacobian(x, val)
                return torch.einsum("bij,bik,bkj->bj", jacobian, matrix, jacobian)
            elif from_diag and to_diag:
                # diag -> diag
                jacobian = self._jacobian(x, val)
                return torch.einsum("bij,bi,bij->bj", jacobian, matrix, jacobian)
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
        b = x1.shape[0]
        if val1 is None:
            val1 = self.forward(x1)
        if val2 is None:
            val2 = self.forward(x2)
        assert val1.shape == val2.shape
        if matrixes is None:
            matrixes = tuple(torch.ones_like(val1).reshape(b, -1) for _ in range(3))
            from_diag = True
        if wrt == "input":
            m11, m12, m22 = matrixes
            jac_1 = self._jacobian(x1, val1)
            jac_2 = self._jacobian(x2, val2)

            if not from_diag and not to_diag:
                # full -> full
                return tuple(
                    torch.einsum("bji,bjk,bkq->biq", jac_i, m, jac_j)
                    for jac_i, m, jac_j in [(jac_1, m11, jac_1), (jac_1, m12, jac_2), (jac_2, m22, jac_2)]
                )
            elif from_diag and not to_diag:
                # diag -> full
                return tuple(
                    torch.einsum("bji,bj,bjk->bik", jac_i, m, jac_j)
                    for jac_i, m, jac_j in [(jac_1, m11, jac_1), (jac_1, m12, jac_2), (jac_2, m22, jac_2)]
                )
            elif not from_diag and to_diag:
                # full -> diag
                return tuple(
                    torch.einsum("bji,bjk,bki->bi", jac_i, m, jac_j)
                    for jac_i, m, jac_j in [(jac_1, m11, jac_1), (jac_1, m12, jac_2), (jac_2, m22, jac_2)]
                )
            elif from_diag and to_diag:
                # diag -> diag
                return tuple(
                    torch.einsum("bji,bj,bji->bi", jac_i, m, jac_j)
                    for jac_i, m, jac_j in [(jac_1, m11, jac_1), (jac_1, m12, jac_2), (jac_2, m22, jac_2)]
                )
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None