import torch
import torch.nn.functional as F
from torch import nn, Tensor
from stochman.nnj.abstract_jacobian import AbstractJacobian

from typing import Optional, Tuple, List, Union

class Upsample(AbstractJacobian, nn.Upsample):
    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        xs = x.shape
        vs = val.shape

        dims1 = tuple(range(1, x.ndim))
        dims2 = tuple(range(-x.ndim + 1, 0))

        return (
            F.interpolate(
                jac_in.movedim(dims1, dims2).reshape(-1, *xs[1:]),
                self.size,
                self.scale_factor,
                self.mode,
                self.align_corners,
            )
            .reshape(xs[0], *jac_in.shape[x.ndim :], *vs[1:])
            .movedim(dims2, dims1)
        )

    def _jvp(self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input") -> Tensor:
        """
        jacobian vector product
        """
        if wrt == "input":
            raise NotImplementedError
        elif wrt == "weight":
            raise NotImplementedError

    def _vjp(self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input") -> Tensor:
        """
        vector jacobian product
        """
        if wrt == "input":
            raise NotImplementedError
        elif wrt == "weight":
            raise NotImplementedError

    def _jmp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        jacobian matrix product
        """
        if wrt == "input":
            raise NotImplementedError
        elif wrt == "weight":
            raise NotImplementedError

    def _mjp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        matrix jacobian product
        """
        if wrt == "input":
            b, c1, h1, w1 = x.shape
            _, c2, h2, w2 = val.shape
            assert c1 == c2
            assert matrix.shape == (b, c2 * h2 * w2, c2 * h2 * w2)

            weight = torch.ones(1, 1, int(self.scale_factor), int(self.scale_factor), device=x.device)

            matrix = matrix.reshape(b, c2, h2 * w2, c2, h2 * w2)
            matrix = matrix.movedim(2, 3)
            matrix_J = F.conv2d(
                matrix.reshape(b * c2 * c2 * h2 * w2, 1, h2, w2),
                weight=weight,
                bias=None,
                stride=int(self.scale_factor),
                padding=0,
                dilation=1,
                groups=1,
            ).reshape(b * c2, c2, h2 * w2, h1 * w1)

            matrix_J = matrix_J.movedim(2, 3)
            return matrix_J.reshape(b, c2 * h2 * w2, c1 * h1 * w1)
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
    ) -> Tensor:
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
                return self._jacobian_wrt_input_sandwich_full_to_full(x, val, matrix)
            elif from_diag and not to_diag:
                # diag -> full
                raise NotImplementedError
            elif not from_diag and to_diag:
                # full -> diag
                raise NotImplementedError
            elif from_diag and to_diag:
                # diag -> diag
                return self._jacobian_wrt_input_sandwich_diag_to_diag(x, val, matrix)
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
    ) -> Union[Tensor, Tuple]:
        """
        jacobian.T matrix jacobian product, computed on 2 inputs (considering cross terms)
        """
        assert x1.shape == x2.shape
        if val1 is None:
            val1 = self.forward(x1)
        if wrt == "input":
            if not from_diag and not to_diag:
                # full -> full
                return tuple(self._jacobian_wrt_input_sandwich_full_to_full(x1, val1, m) for m in matrixes)
            elif from_diag and not to_diag:
                # diag -> full
                raise NotImplementedError
            elif not from_diag and to_diag:
                # full -> diag
                raise NotImplementedError
            elif from_diag and to_diag:
                # diag -> diag
                raise NotImplementedError
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

    def _jacobian_wrt_input_sandwich_full_to_full(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        assert c1 == c2

        weight = torch.ones(1, 1, int(self.scale_factor), int(self.scale_factor), device=x.device)

        tmp = tmp.reshape(b, c2, h2 * w2, c2, h2 * w2)
        tmp = tmp.movedim(2, 3)
        tmp_J = F.conv2d(
            tmp.reshape(b * c2 * c2 * h2 * w2, 1, h2, w2),
            weight=weight,
            bias=None,
            stride=int(self.scale_factor),
            padding=0,
            dilation=1,
            groups=1,
        ).reshape(b * c2 * c2, h2 * w2, h1 * w1)

        Jt_tmpt = tmp_J.movedim(-1, -2)

        Jt_tmpt_J = F.conv2d(
            Jt_tmpt.reshape(b * c2 * c2 * h1 * w1, 1, h2, w2),
            weight=weight,
            bias=None,
            stride=int(self.scale_factor),
            padding=0,
            dilation=1,
            groups=1,
        ).reshape(b * c2 * c2, h1 * w1, h1 * w1)

        Jt_tmp_J = Jt_tmpt_J.movedim(-1, -2)

        Jt_tmp_J = Jt_tmp_J.reshape(b, c2, c2, h1 * w1, h1 * w1)
        Jt_tmp_J = Jt_tmp_J.movedim(2, 3)
        Jt_tmp_J = Jt_tmp_J.reshape(b, c2 * h1 * w1, c2 * h1 * w1)

        return Jt_tmp_J

    def _jacobian_wrt_input_sandwich_diag_to_diag(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        weight = torch.ones(c2, c1, int(self.scale_factor), int(self.scale_factor), device=x.device)

        tmp_diag = F.conv2d(
            tmp_diag.reshape(-1, c2, h2, w2),
            weight=weight,
            bias=None,
            stride=int(self.scale_factor),
            padding=0,
            dilation=1,
            groups=1,
        )

        return tmp_diag.reshape(b, c1 * h1 * w1)
