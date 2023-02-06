import torch
from torch import nn, Tensor
from stochman.nnj.abstract_jacobian import AbstractJacobian
from stochman.nnj.sequential import Sequential

from typing import Optional, Tuple, List, Union

class ResidualBlock(nn.Module):
    def __init__(self, *args, add_hooks: bool = False):
        super().__init__()

        self._F = Sequential(*args, add_hooks=add_hooks)

    def forward(self, x):
        return self._F(x) + x

    def _jvp(
        self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input"
    ) -> Tensor:
        """
        jacobian vector product
        """
        jvp = self._F._jvp(x, None if val is None else val - x, vector, wrt=wrt)
        if wrt == "input":
            return jvp + vector
        elif wrt == "weight":
            return jvp

    def _vjp(
        self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input"
    ) -> Tensor:
        """
        vector jacobian product
        """
        vjp = self._F._vjp(x, None if val is None else val - x, vector, wrt=wrt)
        if wrt == "input":
            return vjp + vector
        elif wrt == "weight":
            return vjp
        
    def _jmp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        jacobian matrix product
        """
        jmp = self._F._jmp(x, None if val is None else val - x, matrix, wrt=wrt)
        if wrt == "input":
            return jmp + matrix
        elif wrt == "weight":
            return jmp

    def _mjp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        matrix jacobian product
        """
        mjp = self._F._mjp(x, None if val is None else val - x, matrix, wrt=wrt)
        if wrt == "input":
            return mjp + matrix
        elif wrt == "weight":
            return mjp

    def _jTmjp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, Tuple]:
        """
        jacobian.T matrix jacobian product
        """
        # TODO: deal with diagonal matrix
        if val is None:
            raise NotImplementedError
        if matrix is None:
            raise NotImplementedError
        jTmjp = self._F._jTmjp(
            x,
            None if val is None else val - x,
            matrix,
            wrt=wrt,
            from_diag=from_diag,
            to_diag=to_diag,
            diag_backprop=diag_backprop,
        )
        if wrt == "input":
            if diag_backprop:
                return jTmjp + matrix
            mjp = self._F._mjp(x, None if val is None else val - x, matrix, wrt=wrt)
            jTmp = self._F._mjp(
                x, None if val is None else val - x, matrix.transpose(1, 2), wrt=wrt
            ).transpose(1, 2)
            return jTmjp + mjp + jTmp + matrix
        elif wrt == "weight":
            return jTmjp

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
    ) -> Union[Tensor, Tuple]:
        """
        jacobian.T matrix jacobian product, computed on 2 inputs (considering cross terms)
        """
        assert x1.shape == x2.shape
        # TODO: deal with diagonal matrix
        if val1 is None:
            raise NotImplementedError
        if val2 is None:
            raise NotImplementedError
        if matrixes is None:
            raise NotImplementedError
        if from_diag or diag_backprop:
            raise NotImplementedError
        jTmjps = self._F._jTmjp_batch2(
            x1,
            x2,
            None if val1 is None else val1 - x1,
            None if val2 is None else val2 - x2,
            matrixes,
            wrt=wrt,
            from_diag=from_diag,
            to_diag=to_diag,
            diag_backprop=diag_backprop,
        )
        if wrt == "input":
            if to_diag:
                raise NotImplementedError
            m11, m12, m22 = matrixes
            mjps = tuple(
                self._F._mjp(x_i, None if val_i is None else val_i - x_i, m, wrt=wrt)
                for x_i, val_i, m in [(x1, val1, m11), (x2, val2, m12), (x2, val2, m22)]
            )
            jTmps = tuple(
                self._F._mjp(
                    x_i, None if val_i is None else val_i - x_i, m.transpose(1, 2), wrt=wrt
                ).transpose(1, 2)
                for x_i, val_i, m in [(x1, val1, m11), (x1, val1, m12), (x2, val2, m22)]
            )
            # new_m11 = J1T * m11 * J1 + m11 * J1 + J1T * m11 + m11
            # new_m12 = J1T * m12 * J2 + m12 * J2 + J1T * m12 + m12
            # new_m22 = J2T * m22 * J2 + m22 * J2 + J2T * m22 + m22
            return tuple(jTmjp + mjp + jTmp + m for jTmjp, mjp, jTmp, m in zip(jTmjps, mjps, jTmps, matrixes))
        elif wrt == "weight":
            return jTmjps
