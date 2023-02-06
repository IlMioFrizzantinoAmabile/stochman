import torch
import torch.nn.functional as F
from torch import nn, Tensor
from stochman.nnj.abstract_jacobian import AbstractJacobian

from typing import Optional, Tuple, List, Union

class MaxPool2d(AbstractJacobian, nn.MaxPool2d):
    def forward(self, input: Tensor):
        val, idx = F.max_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            return_indices=True,
        )
        self.idx = idx
        return val

    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        jac_in_orig_shape = jac_in.shape
        jac_in = jac_in.reshape(-1, h1 * w1, *jac_in_orig_shape[4:])
        arange_repeated = torch.repeat_interleave(torch.arange(b * c1), h2 * w2).long()
        idx = self.idx.reshape(-1)
        jac_in = jac_in[arange_repeated, idx, :, :, :].reshape(*val.shape, *jac_in_orig_shape[4:])
        return jac_in

    def _jvp(
        self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input"
    ) -> Tensor:
        """
        jacobian vector product
        """
        if wrt == "input":
            raise NotImplementedError
        elif wrt == "weight":
            return None

    def _vjp(
        self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input"
    ) -> Tensor:
        """
        vector jacobian product
        """
        if wrt == "input":
            raise NotImplementedError
        elif wrt == "weight":
            return None

    def _jmp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        jacobian matrix product
        """
        if wrt == "input":
            if val is None:
                val = self.forward(x)
            b, c1, h1, w1 = x.shape
            c2, h2, w2 = val.shape[1:]

            matrix_orig_shape = matrix.shape
            matrix = matrix.reshape(-1, h1 * w1, *matrix_orig_shape[4:])
            arange_repeated = torch.repeat_interleave(torch.arange(b * c1), h2 * w2).long()
            idx = self.idx.reshape(-1)
            matrix = matrix[arange_repeated, idx, :, :, :].reshape(*val.shape, *matrix_orig_shape[4:])
            return matrix
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

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

            matrix = (
                matrix.reshape(b, c1, h2 * w2, c1, h2 * w2)
                .movedim(-2, -3)
                .reshape(b * c1 * c1, h2 * w2, h2 * w2)
            )
            # indexes for batch, channel and row
            arange_repeated = torch.repeat_interleave(torch.arange(b * c1 * c1 * h2 * w2), h2 * w2).long()
            arange_repeated = arange_repeated.reshape(b * c1 * c1 * h2 * w2, h2 * w2)
            # indexes for col
            idx = self.idx.reshape(b, c1, h2 * w2).unsqueeze(2).expand(-1, -1, h2 * w2, -1)
            idx_col = idx.unsqueeze(1).expand(-1, c1, -1, -1, -1).reshape(b * c1 * c1 * h2 * w2, h2 * w2)

            matrix_J = torch.zeros((b * c1 * c1, h2 * w2, h1 * w1), device=matrix.device)
            matrix_J[arange_repeated, idx_col] = matrix
            matrix_J = (
                matrix_J.reshape(b, c1, c1, h1 * w1, h1 * w1)
                .movedim(-2, -3)
                .reshape(b, c1 * h1 * w1, c1 * h1 * w1)
            )

            return matrix_J
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
        if val1 is None or val2 is None:
            raise NotImplementedError
        if wrt == "input":
            m11, m12, m22 = matrixes
            self.forward(x1)
            idx1 = self.idx
            self.forward(x2)
            idx2 = self.idx
            if not from_diag and not to_diag:
                # full -> full
                return tuple(
                    self._jacobian_wrt_input_sandwich_full_to_full_batch2(self, x1, val1, idx_left, idx_right, m)
                    for idx_left, m, idx_right in [(idx1, m11, idx1), (idx1, m12, idx2), (idx1, m22, idx2)]
                )
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

        tmp = tmp.reshape(b, c1, h2 * w2, c1, h2 * w2).movedim(-2, -3).reshape(b * c1 * c1, h2 * w2, h2 * w2)
        Jt_tmp_J = torch.zeros((b * c1 * c1, h1 * w1, h1 * w1), device=tmp.device)
        # indexes for batch and channel
        arange_repeated = torch.repeat_interleave(torch.arange(b * c1 * c1), h2 * w2 * h2 * w2).long()
        arange_repeated = arange_repeated.reshape(b * c1 * c1, h2 * w2, h2 * w2)
        # indexes for height and width
        idx = self.idx.reshape(b, c1, h2 * w2).unsqueeze(2).expand(-1, -1, h2 * w2, -1)
        idx_col = idx.unsqueeze(1).expand(-1, c1, -1, -1, -1).reshape(b * c1 * c1, h2 * w2, h2 * w2)
        idx_row = (
            idx.unsqueeze(2).expand(-1, -1, c1, -1, -1).reshape(b * c1 * c1, h2 * w2, h2 * w2).movedim(-1, -2)
        )

        Jt_tmp_J[arange_repeated, idx_row, idx_col] = tmp
        Jt_tmp_J = (
            Jt_tmp_J.reshape(b, c1, c1, h1 * w1, h1 * w1)
            .movedim(-2, -3)
            .reshape(b, c1 * h1 * w1, c1 * h1 * w1)
        )

        return Jt_tmp_J

    def _jacobian_wrt_input_sandwich_diag_to_diag(self, x: Tensor, val: Tensor, diag_tmp: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        new_tmp = torch.zeros_like(x)
        new_tmp = new_tmp.reshape(b * c1, h1 * w1)

        # indexes for batch and channel
        arange_repeated = torch.repeat_interleave(torch.arange(b * c1), h2 * w2).long()
        arange_repeated = arange_repeated.reshape(b * c2, h2 * w2)
        # indexes for height and width
        idx = self.idx.reshape(b * c2, h2 * w2)

        new_tmp[arange_repeated, idx] = diag_tmp.reshape(b * c2, h2 * w2)

        return new_tmp.reshape(b, c1 * h1 * w1)

    def _jacobian_wrt_input_sandwich_full_to_full_batch2(self, x, val, idx_left, idx_right, matrix: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]
        assert c1 == c2

        matrix = matrix.reshape(b, c1, h2 * w2, c1, h2 * w2).movedim(-2, -3).reshape(b * c1 * c1, h2 * w2, h2 * w2)
        Jt_matrix_J = torch.zeros((b * c1 * c1, h1 * w1, h1 * w1), device=matrix.device)
        # indexes for batch and channel
        arange_repeated = torch.repeat_interleave(torch.arange(b * c1 * c1), h2 * w2 * h2 * w2).long()
        arange_repeated = arange_repeated.reshape(b * c1 * c1, h2 * w2, h2 * w2)
        # indexes for height and width
        idx_left = idx_left.reshape(b, c1, h2 * w2).unsqueeze(2).expand(-1, -1, h2 * w2, -1)
        idx_right = idx_right.reshape(b, c1, h2 * w2).unsqueeze(2).expand(-1, -1, h2 * w2, -1)
        idx_row = (
            idx_left.unsqueeze(2).expand(-1, -1, c1, -1, -1).reshape(b * c1 * c1, h2 * w2, h2 * w2).movedim(-1, -2)
        )
        idx_col = idx_right.unsqueeze(1).expand(-1, c1, -1, -1, -1).reshape(b * c1 * c1, h2 * w2, h2 * w2)

        Jt_matrix_J[arange_repeated, idx_row, idx_col] = matrix
        Jt_matrix_J = (
            Jt_matrix_J.reshape(b, c1, c1, h1 * w1, h1 * w1)
            .movedim(-2, -3)
            .reshape(b, c1 * h1 * w1, c1 * h1 * w1)
        )

        return Jt_matrix_J