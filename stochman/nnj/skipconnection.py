import torch
from torch import nn, Tensor
from stochman.nnj.abstract_jacobian import AbstractJacobian
from stochman.nnj.sequential import Sequential

from typing import Optional, Tuple, List, Union

class SkipConnection(AbstractJacobian, nn.Module):
    def __init__(self, *args, add_hooks: bool = False):
        super().__init__()

        self._F = Sequential(*args, add_hooks=add_hooks)

    def forward(self, x):
        return torch.cat([x, self._F(x)], dim=1)

    def _jvp(
        self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input"
    ) -> Tensor:
        """
        jacobian vector product
        """
        if wrt == "input":
            raise NotImplementedError
        elif wrt == "weight":
            raise NotImplementedError

    def _vjp(
        self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input"
    ) -> Tensor:
        """
        vector jacobian product
        """
        b, l = x.shape
        vjp = self._F._vjp(x, None if val is None else val[:, l:], vector[:, l:], wrt=wrt)
        if wrt == "input":
            return vector[:, :l] + vjp
        elif wrt == "weight":
            return vjp

    def _jmp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        jacobian matrix product
        """
        if wrt == "input":
            jmp = self._F._jmp(x, val, matrix, wrt=wrt)
            return torch.cat([matrix, jmp], dim=1)
        elif wrt == "weight":
            raise NotImplementedError

    def _mjp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        matrix jacobian product
        """
        b, l = x.shape
        mjp = self._F._mjp(x, None if val is None else val[:, l:], matrix[:, :, l:], wrt=wrt)
        if wrt == "input":
            return matrix[:, :, :l] + mjp
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
    ) -> Tensor:
        """
        jacobian.T matrix jacobian product
        """
        assert (not diag_backprop) or (diag_backprop and from_diag and to_diag)
        b, l = x.shape
        jTmjp = self._F._jTmjp(
            x,
            None if val is None else val[:, l:],
            matrix[:, l:, l:] if not from_diag else matrix[:, l:],
            wrt=wrt,
            from_diag=from_diag,
            to_diag=to_diag,
            diag_backprop=diag_backprop,
        )
        if wrt == "input":
            if diag_backprop:
                return jTmjp + matrix[:, :l]
            mjp = self._F._mjp(x, None if val is None else val[:, l:], matrix[:, :l, l:], wrt=wrt)
            jTmp = self._F._mjp(
                x, None if val is None else val[:, l:], matrix[:, l:, :l].transpose(1, 2), wrt=wrt
            ).transpose(1, 2)
            return jTmjp + mjp + jTmp + matrix[:, :l, :l]
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
        b, l = x1.shape
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
            None if val1 is None else val1[:, l:],
            None if val2 is None else val2[:, l:],
            tuple(m[:, l:, l:] for m in matrixes),
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
                self._F._mjp(x_i, None if val_i is None else val_i[:, l:], m[:, :l, l:], wrt=wrt)
                for x_i, val_i, m in [(x1, val1, m11), (x2, val2, m12), (x2, val2, m22)]
            )
            jTmps = tuple(
                self._F._mjp(
                    x_i, None if val_i is None else val_i[:, l:], m[:, l:, :l].transpose(1, 2), wrt=wrt
                ).transpose(1, 2)
                for x_i, val_i, m in [(x1, val1, m11), (x1, val1, m12), (x2, val2, m22)]
            )
            # schematic of the update rule with jacobian products (neglecting batch size)
            # new_m11 = J1T * m11[l:,l:] * J1 + m11[l:,:l] * J1 + J1T * m11[:l,l:] + m11[:l,:l]
            # new_m12 = J1T * m12[l:,l:] * J2 + m12[l:,:l] * J2 + J1T * m12[:l,l:] + m12[:l,:l]
            # new_m22 = J2T * m22[l:,l:] * J2 + m22[l:,:l] * J2 + J2T * m22[:l,l:] + m22[:l,:l]
            return tuple(
                jTmjp + mjp + jTmp + m[:, :l, :l]
                for jTmjp, mjp, jTmp, m in zip(jTmjps, mjps, jTmps, matrixes)
            )
        elif wrt == "weight":
            return jTmjps