import torch
import torch.nn.functional as F
from torch import nn, Tensor
from stochman.nnj.abstract_jacobian import AbstractJacobian

from typing import Optional, Tuple, List, Union


def compute_reversed_padding(padding, kernel_size=1):
    return kernel_size - 1 - padding


class Conv2d(AbstractJacobian, nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode
        )

        dw_padding_h = compute_reversed_padding(self.padding[0], kernel_size=self.kernel_size[0])
        dw_padding_w = compute_reversed_padding(self.padding[1], kernel_size=self.kernel_size[1])
        self.dw_padding = (dw_padding_h, dw_padding_w)

    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        return (
            F.conv2d(
                jac_in.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, c1, h1, w1),
                weight=self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            .reshape(b, *jac_in.shape[4:], c2, h2, w2)
            .movedim((-3, -2, -1), (1, 2, 3))
        )

    def _jacobian_wrt_input(self, x: Tensor, val: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        output_identity = torch.eye(c1 * h1 * w1).unsqueeze(0).expand(b, -1, -1)
        output_identity = output_identity.reshape(b, c1, h1, w1, c1 * h1 * w1)

        # convolve each column
        jacobian = self._jacobian_wrt_input_mult_left_vec(x, val, output_identity)

        # reshape as a (num of output)x(num of input) matrix, one for each batch size
        jacobian = jacobian.reshape(b, c2 * h2 * w2, c1 * h1 * w1)

        return jacobian

    def _jacobian_wrt_weight(self, x: Tensor, val: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        kernel_h, kernel_w = self.kernel_size

        output_identity = torch.eye(c2 * c1 * kernel_h * kernel_w)
        # expand rows as [(input channels)x(kernel height)x(kernel width)] cubes, one for each output channel
        output_identity = output_identity.reshape(c2, c1, kernel_h, kernel_w, c2 * c1 * kernel_h * kernel_w)

        reversed_inputs = torch.flip(x, [-2, -1]).movedim(0, 1)

        # convolve each base element and compute the jacobian
        jacobian = (
            F.conv_transpose2d(
                output_identity.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, c1, kernel_h, kernel_w),
                weight=reversed_inputs,
                bias=None,
                stride=self.stride,
                padding=self.dw_padding,
                dilation=self.dilation,
                groups=self.groups,
                output_padding=0,
            )
            .reshape(c2, *output_identity.shape[4:], b, h2, w2)
            .movedim((-3, -2, -1), (1, 2, 3))
        )

        # transpose the result in (output height)x(output width)
        jacobian = torch.flip(jacobian, [-3, -2])
        # switch batch size and output channel
        jacobian = jacobian.movedim(0, 1)
        # reshape as a (num of output)x(num of weights) matrix, one for each batch size
        jacobian = jacobian.reshape(b, c2 * h2 * w2, c2 * c1 * kernel_h * kernel_w)
        return jacobian

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
            return self._jacobian_wrt_input_mult_left(x, val, vector.unsqueeze(1)).squeeze(1)
        elif wrt == "weight":
            if self.bias is None:
                return self._jacobian_wrt_weight_mult_left(x, val, vector.unsqueeze(1)).squeeze(1)
            else:
                b_term = torch.einsum("bchw->bc", vector.reshape(val.shape))
                return torch.cat(
                    [self._jacobian_wrt_weight_mult_left(x, val, vector.unsqueeze(1)).squeeze(1), b_term],
                    dim=1,
                )

    def _jmp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        jacobian matrix product
        """
        if wrt == "input":
            b, c1, h1, w1 = x.shape
            c2, h2, w2 = val.shape[1:]
            assert matrix.shape[1] == c1 * h1 * w1
            n_col = matrix.shape[2]
            return (
                F.conv2d(
                    matrix.movedim((1), (-1)).reshape(-1, c1, h1, w1),
                    weight=self.weight,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                .reshape(b, n_col, c2 * h2 * w2)
                .movedim((1), (-1))
            )
        elif wrt == "weight":
            raise NotImplementedError

    def _mjp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        matrix jacobian product
        """
        if wrt == "input":
            return self._jacobian_wrt_input_mult_left(x, val, matrix)
        elif wrt == "weight":
            if self.bias is None:
                return self._jacobian_wrt_weight_mult_left(x, val, matrix)
            else:
                b, c, h, w = val.shape
                b_term = torch.einsum("bvchw->bvc", matrix.reshape(b, -1, c, h, w))
                return torch.cat([self._jacobian_wrt_weight_mult_left(x, val, matrix), b_term], dim=2)

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
            if not from_diag and not to_diag:
                # full -> full
                if self.bias is None:
                    return self._jacobian_wrt_weight_sandwich_full_to_full(x, val, matrix)
                else:
                    matrix = self._mjp(x, val, matrix, wrt=wrt)
                    matrix = matrix.movedim(-2, -1)
                    matrix = self._mjp(x, val, matrix, wrt=wrt)
                    matrix = matrix.movedim(-2, -1)
                    return matrix
            elif from_diag and not to_diag:
                # diag -> full
                raise NotImplementedError
            elif not from_diag and to_diag:
                # full -> diag
                if self.bias is None:
                    return self._jacobian_wrt_weight_sandwich_full_to_diag(x, val, matrix)
                else:
                    # TODO: Implement this in a smarter way
                    return torch.diagonal(
                        self._jTmjp(x, val, matrix, wrt=wrt, from_diag=from_diag, to_diag=False),
                        dim1=1,
                        dim2=2,
                    )
            elif from_diag and to_diag:
                # diag -> diag
                return self._jacobian_wrt_weight_sandwich_diag_to_diag(x, val, matrix)

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
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        jacobian.T matrix jacobian product, computed on 2 inputs (considering cross terms)
        """
        assert x1.shape == x2.shape
        if val1 is None or val2 is None:
            raise NotImplementedError
        if wrt == "input":
            if not from_diag and not to_diag:
                # full -> full
                return tuple(
                    self._jacobian_wrt_input_sandwich_full_to_full(x1, val1, m) for m in matrixes
                )  # not dependent on x1,val1, only on their shape
            elif from_diag and not to_diag:
                # diag -> full
                raise NotImplementedError
            elif not from_diag and to_diag:
                # full -> diag
                raise NotImplementedError
            elif from_diag and to_diag:
                # diag -> diag
                return tuple(
                    self._jacobian_wrt_input_sandwich_diag_to_diag(x1, val1, m) for m in matrixes
                )  # not dependent on x1,val1, only on their shape
        elif wrt == "weight":
            m11, m12, m22 = matrixes
            if not from_diag and not to_diag:
                # full -> full
                if self.bias is None:
                    return tuple(
                        self._jacobian_wrt_weight_T_mult_right(
                            x_i, val_i, self._jacobian_wrt_weight_mult_left(x_j, val_j, m)
                        )
                        for x_i, val_i, m, x_j, val_j in [
                            (x1, val1, m11, x1, val1),
                            (x1, val1, m12, x2, val2),
                            (x2, val2, m22, x2, val2),
                        ]
                    )
                else:
                    return tuple(
                        self._mjp(
                            x_i, val_i, self._mjp(x_j, val_j, m, wrt=wrt).movedim(-2, -1), wrt=wrt
                        ).movedim(-2, -1)
                        for x_i, val_i, m, x_j, val_j in [
                            (x1, val1, m11, x1, val1),
                            (x1, val1, m12, x2, val2),
                            (x2, val2, m22, x2, val2),
                        ]
                    )
            elif from_diag and not to_diag:
                # diag -> full
                raise NotImplementedError
            elif not from_diag and to_diag:
                # full -> diag
                # TODO: Implement this in a smarter way
                if self.bias is None:
                    return tuple(
                        torch.diagonal(
                            self._jacobian_wrt_weight_T_mult_right(
                                x_i, val_i, self._jacobian_wrt_weight_mult_left(x_j, val_j, m)
                            ),
                            dim1=1,
                            dim2=2,
                        )
                        for x_i, val_i, m, x_j, val_j in [
                            (x1, val1, m11, x1, val1),
                            (x1, val1, m12, x2, val2),
                            (x2, val2, m22, x2, val2),
                        ]
                    )
                else:
                    return tuple(
                        torch.diagonal(
                            self._mjp(
                                x_i, val_i, self._mjp(x_j, val_j, m, wrt=wrt).movedim(-2, -1), wrt=wrt
                            ).movedim(-2, -1),
                            dim1=1,
                            dim2=2,
                        )
                        for x_i, val_i, m, x_j, val_j in [
                            (x1, val1, m11, x1, val1),
                            (x1, val1, m12, x2, val2),
                            (x2, val2, m22, x2, val2),
                        ]
                    )
            elif from_diag and to_diag:
                # diag -> diag
                if self.bias is None:
                    return tuple(
                        (
                            self._jacobian_wrt_weight_sandwich_diag_to_diag(x1, val1, m11),
                            self._jacobian_wrt_weight_sandwich_diag_to_diag_multipoint(
                                x1, x2, val1, val2, m12
                            ),
                            self._jacobian_wrt_weight_sandwich_diag_to_diag(x2, val2, m22),
                        )
                    )
                else:
                    raise NotImplementedError

    def _jacobian_sandwich(
        self, x: Tensor, val: Tensor, tmp, wrt="input", diag_inp: bool = True, diag_out: bool = True
    ):
        if wrt == "input":
            if not diag_inp and not diag_out:
                # full -> full
                return self._jacobian_wrt_input_sandwich_full_to_full(x, val, tmp)
            elif diag_inp and not diag_out:
                # diag -> full
                raise NotImplementedError
            elif not diag_inp and diag_out:
                # full -> diag
                raise NotImplementedError
            elif diag_inp and diag_out:
                # diag -> diag
                return self._jacobian_wrt_input_sandwich_diag_to_diag(x, val, tmp)
        elif wrt == "weight":
            if not diag_inp and not diag_out:
                # full -> full
                return self._jacobian_wrt_weight_sandwich_full_to_full(x, val, tmp)
            elif diag_inp and not diag_out:
                # diag -> full
                raise NotImplementedError
            elif not diag_inp and diag_out:
                # full -> diag
                return self._jacobian_wrt_weight_sandwich_full_to_diag(x, val, tmp)
            elif diag_inp and diag_out:
                # diag -> diag
                return self._jacobian_wrt_weight_sandwich_diag_to_diag(x, val, tmp)

    def _jacobian_sandwich_multipoint(
        self,
        x1: Tensor,
        x2: Tensor,
        val1: Tensor,
        val2: Tensor,
        tmps: Tuple[Tensor, Tensor, Tensor],
        wrt="input",
        diag_inp: bool = True,
        diag_out: bool = True,
    ):
        if wrt == "input":
            if not diag_inp and not diag_out:
                # full -> full
                return tuple(
                    self._jacobian_wrt_input_sandwich_full_to_full(x1, val1, tmp_diag) for tmp_diag in tmps
                )  # not dependent on x1,val1, only on their shape
            elif diag_inp and not diag_out:
                # diag -> full
                raise NotImplementedError
            elif not diag_inp and diag_out:
                # full -> diag
                raise NotImplementedError
            elif diag_inp and diag_out:
                # diag -> diag
                return tuple(
                    self._jacobian_wrt_input_sandwich_diag_to_diag(x1, val1, tmp_diag) for tmp_diag in tmps
                )  # not dependent on x1,val1, only on their shape
        elif wrt == "weight":
            if not diag_inp and not diag_out:
                # full -> full
                tmp11, tmp12, tmp22 = tmps
                tmp11 = self._jacobian_wrt_weight_T_mult_right(
                    x1, val1, self._jacobian_wrt_weight_mult_left(x1, val1, tmp11)
                )
                tmp12 = self._jacobian_wrt_weight_T_mult_right(
                    x1, val1, self._jacobian_wrt_weight_mult_left(x2, val2, tmp12)
                )
                tmp22 = self._jacobian_wrt_weight_T_mult_right(
                    x2, val2, self._jacobian_wrt_weight_mult_left(x2, val2, tmp22)
                )
                return (tmp11, tmp12, tmp22)
            elif diag_inp and not diag_out:
                # diag -> full
                raise NotImplementedError
            elif not diag_inp and diag_out:
                # full -> diag
                # TODO: Implement this in a smarter way
                tmp11, tmp12, tmp22 = tmps
                tmp11 = self._jacobian_wrt_weight_T_mult_right(
                    x1, val1, self._jacobian_wrt_weight_mult_left(x1, val1, tmp11)
                )
                tmp12 = self._jacobian_wrt_weight_T_mult_right(
                    x1, val1, self._jacobian_wrt_weight_mult_left(x2, val2, tmp12)
                )
                tmp22 = self._jacobian_wrt_weight_T_mult_right(
                    x2, val2, self._jacobian_wrt_weight_mult_left(x2, val2, tmp22)
                )
                return tuple(torch.diagonal(tmp, dim1=1, dim2=2) for tmp in [tmp11, tmp12, tmp22])
            elif diag_inp and diag_out:
                # diag -> diag
                diag_tmp11, diag_tmp12, diag_tmp22 = tmps
                diag_tmp11 = self._jacobian_wrt_weight_sandwich_diag_to_diag(x1, val1, diag_tmp11)
                diag_tmp12 = self._jacobian_wrt_weight_sandwich_diag_to_diag_multipoint(
                    x1, val1, x2, val2, diag_tmp12
                )
                diag_tmp22 = self._jacobian_wrt_weight_sandwich_diag_to_diag(x2, val2, diag_tmp22)
                return (diag_tmp11, diag_tmp12, diag_tmp22)

    def _jacobian_wrt_input_T_mult_right(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        num_of_cols = tmp.shape[-1]
        assert list(tmp.shape) == [b, c2 * h2 * w2, num_of_cols]
        # expand rows as cubes [(output channel)x(output height)x(output width)]
        tmp = tmp.reshape(b, c2, h2, w2, num_of_cols)

        # convolve each column
        Jt_tmp = (
            F.conv_transpose2d(
                tmp.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, c2, h2, w2),
                weight=self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                output_padding=self.output_padding,
            )
            .reshape(b, *tmp.shape[4:], c1, h1, w1)
            .movedim((-3, -2, -1), (1, 2, 3))
        )

        # reshape as a (num of input)x(num of column) matrix, one for each batch size
        Jt_tmp = Jt_tmp.reshape(b, c1 * h1 * w1, num_of_cols)
        return Jt_tmp

    def _jacobian_wrt_input_mult_left(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        num_of_rows = tmp.shape[-2]
        assert list(tmp.shape) == [b, num_of_rows, c2 * h2 * w2]
        # expand rows as cubes [(output channel)x(output height)x(output width)]
        tmp_rows = tmp.movedim(-1, -2).reshape(b, c2, h2, w2, num_of_rows)
        # see rows as columns of the transposed matrix
        tmpt_cols = tmp_rows

        # convolve each column
        Jt_tmptt_cols = (
            F.conv_transpose2d(
                tmpt_cols.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, c2, h2, w2),
                weight=self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                output_padding=self.output_padding,
            )
            .reshape(b, *tmpt_cols.shape[4:], c1, h1, w1)
            .movedim((-3, -2, -1), (1, 2, 3))
        )

        # reshape as a (num of input)x(num of output) matrix, one for each batch size
        Jt_tmptt_cols = Jt_tmptt_cols.reshape(b, c1 * h1 * w1, num_of_rows)

        # transpose
        tmp_J = Jt_tmptt_cols.movedim(1, 2)
        return tmp_J

    def _jacobian_wrt_weight_T_mult_right(
        self, x: Tensor, val: Tensor, tmp: Tensor, use_less_memory: bool = True
    ) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        kernel_h, kernel_w = self.kernel_size

        num_of_cols = tmp.shape[-1]

        # expand rows as cubes [(output channel)x(output height)x(output width)]
        tmp = tmp.reshape(b, c2, h2, w2, num_of_cols)
        # transpose the images in (output height)x(output width)
        tmp = torch.flip(tmp, [-3, -2])
        # switch batch size and output channel
        tmp = tmp.movedim(0, 1)

        if use_less_memory:
            # define moving sum for Jt_tmp
            Jt_tmp = torch.zeros(b, c2 * c1 * kernel_h * kernel_w, num_of_cols, device=x.device)
            for i in range(b):
                # set the weight to the convolution
                input_single_batch = x[i : i + 1, :, :, :]
                reversed_input_single_batch = torch.flip(input_single_batch, [-2, -1]).movedim(0, 1)

                tmp_single_batch = tmp[:, i : i + 1, :, :, :]

                # convolve each column
                Jt_tmp_single_batch = (
                    F.conv2d(
                        tmp_single_batch.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, 1, h2, w2),
                        weight=reversed_input_single_batch,
                        bias=None,
                        stride=self.stride,
                        padding=self.dw_padding,
                        dilation=self.dilation,
                        groups=self.groups,
                    )
                    .reshape(c2, *tmp_single_batch.shape[4:], c1, kernel_h, kernel_w)
                    .movedim((-3, -2, -1), (1, 2, 3))
                )

                # reshape as a (num of weights)x(num of column) matrix
                Jt_tmp_single_batch = Jt_tmp_single_batch.reshape(c2 * c1 * kernel_h * kernel_w, num_of_cols)
                Jt_tmp[i, :, :] = Jt_tmp_single_batch

        else:
            reversed_inputs = torch.flip(x, [-2, -1]).movedim(0, 1)

            # convolve each column
            Jt_tmp = (
                F.conv2d(
                    tmp.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, b, h2, w2),
                    weight=reversed_inputs,
                    bias=None,
                    stride=self.stride,
                    padding=self.dw_padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                .reshape(c2, *tmp.shape[4:], c1, kernel_h, kernel_w)
                .movedim((-3, -2, -1), (1, 2, 3))
            )

            # reshape as a (num of weights)x(num of column) matrix
            Jt_tmp = Jt_tmp.reshape(c2 * c1 * kernel_h * kernel_w, num_of_cols)

        return Jt_tmp

    def _jacobian_wrt_weight_mult_left(
        self, x: Tensor, val: Tensor, tmp: Tensor, use_less_memory: bool = True
    ) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]
        kernel_h, kernel_w = self.kernel_size
        num_of_rows = tmp.shape[-2]

        # expand rows as cubes [(output channel)x(output height)x(output width)]
        tmp_rows = tmp.movedim(-1, -2).reshape(b, c2, h2, w2, num_of_rows)
        # see rows as columns of the transposed matrix
        tmpt_cols = tmp_rows
        # transpose the images in (output height)x(output width)
        tmpt_cols = torch.flip(tmpt_cols, [-3, -2])
        # switch batch size and output channel
        tmpt_cols = tmpt_cols.movedim(0, 1)

        if use_less_memory:

            tmp_J = torch.zeros(b, c2 * c1 * kernel_h * kernel_w, num_of_rows, device=x.device)
            for i in range(b):
                # set the weight to the convolution
                input_single_batch = x[i : i + 1, :, :, :]
                reversed_input_single_batch = torch.flip(input_single_batch, [-2, -1]).movedim(0, 1)

                tmp_single_batch = tmpt_cols[:, i : i + 1, :, :, :]

                # convolve each column
                tmp_J_single_batch = (
                    F.conv2d(
                        tmp_single_batch.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, 1, h2, w2),
                        weight=reversed_input_single_batch,
                        bias=None,
                        stride=self.stride,
                        padding=self.dw_padding,
                        dilation=self.dilation,
                        groups=self.groups,
                    )
                    .reshape(c2, *tmp_single_batch.shape[4:], c1, kernel_h, kernel_w)
                    .movedim((-3, -2, -1), (1, 2, 3))
                )

                # reshape as a (num of weights)x(num of column) matrix
                tmp_J_single_batch = tmp_J_single_batch.reshape(c2 * c1 * kernel_h * kernel_w, num_of_rows)
                tmp_J[i, :, :] = tmp_J_single_batch

            # transpose
            tmp_J = tmp_J.movedim(-1, -2)
        else:
            # set the weight to the convolution
            reversed_inputs = torch.flip(x, [-2, -1]).movedim(0, 1)

            # convolve each column
            Jt_tmptt_cols = (
                F.conv2d(
                    tmp_single_batch.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, b, h2, w2),
                    weight=reversed_inputs,
                    bias=None,
                    stride=self.stride,
                    padding=self.dw_padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                .reshape(c2, *tmp_single_batch.shape[4:], c1, kernel_h, kernel_w)
                .movedim((-3, -2, -1), (1, 2, 3))
            )

            # reshape as a (num of input)x(num of output) matrix, one for each batch size
            Jt_tmptt_cols = Jt_tmptt_cols.reshape(c2 * c1 * kernel_h * kernel_w, num_of_rows)
            # transpose
            tmp_J = Jt_tmptt_cols.movedim(0, 1)

        return tmp_J

    def _jacobian_wrt_input_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if not diag_inp and not diag_out:
            return self._jacobian_wrt_input_sandwich_full_to_full(x, val, tmp)
        elif not diag_inp and diag_out:
            return self._jacobian_wrt_input_sandwich_full_to_diag(x, val, tmp)
        elif diag_inp and not diag_out:
            return self._jacobian_wrt_input_sandwich_diag_to_full(x, val, tmp)
        elif diag_inp and diag_out:
            return self._jacobian_wrt_input_sandwich_diag_to_diag(x, val, tmp)

    def _jacobian_wrt_weight_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if not diag_inp and not diag_out:
            return self._jacobian_wrt_weight_sandwich_full_to_full(x, val, tmp)
        elif not diag_inp and diag_out:
            return self._jacobian_wrt_weight_sandwich_full_to_diag(x, val, tmp)
        elif diag_inp and not diag_out:
            return self._jacobian_wrt_weight_sandwich_diag_to_full(x, val, tmp)
        elif diag_inp and diag_out:
            return self._jacobian_wrt_weight_sandwich_diag_to_diag(x, val, tmp)

    def _jacobian_wrt_input_sandwich_full_to_full(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        return self._jacobian_wrt_input_mult_left(x, val, self._jacobian_wrt_input_T_mult_right(x, val, tmp))

    def _jacobian_wrt_input_sandwich_full_to_diag(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        raise NotImplementedError

    def _jacobian_wrt_input_sandwich_diag_to_full(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:
        raise NotImplementedError

    def _jacobian_wrt_input_sandwich_diag_to_diag(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        _, c2, h2, w2 = val.shape

        input_tmp = tmp_diag.reshape(b, c2, h2, w2)

        output_tmp = (
            F.conv_transpose2d(
                input_tmp.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, c2, h2, w2),
                weight=self.weight**2,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                output_padding=0,
            )
            .reshape(b, *input_tmp.shape[4:], c1, h1, w1)
            .movedim((-3, -2, -1), (1, 2, 3))
        )

        diag_Jt_tmp_J = output_tmp.reshape(b, c1 * h1 * w1)
        return diag_Jt_tmp_J

    def _jacobian_wrt_weight_sandwich_full_to_full(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        return self._jacobian_wrt_weight_mult_left(
            x, val, self._jacobian_wrt_weight_T_mult_right(x, val, tmp)
        )

    def _jacobian_wrt_weight_sandwich_full_to_diag(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        # TODO: Implement this in a smarter way
        return torch.diagonal(self._jacobian_wrt_weight_sandwich_full_to_full(x, val, tmp), dim1=1, dim2=2)

    def _jacobian_wrt_weight_sandwich_diag_to_full(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:
        raise NotImplementedError

    def _jacobian_wrt_weight_sandwich_diag_to_diag(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:

        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]
        _, _, kernel_h, kernel_w = self.weight.shape

        input_tmp = tmp_diag.reshape(b, c2, h2, w2)
        # transpose the images in (output height)x(output width)
        input_tmp = torch.flip(input_tmp, [-3, -2, -1])
        # switch batch size and output channel
        input_tmp = input_tmp.movedim(0, 1)

        # define moving sum for Jt_tmp
        output_tmp = torch.zeros(b, c2 * c1 * kernel_h * kernel_w, device=x.device)
        flip_squared_input = torch.flip(x, [-3, -2, -1]).movedim(0, 1) ** 2

        for i in range(b):
            # set the weight to the convolution
            weigth_sq = flip_squared_input[:, i : i + 1, :, :]
            input_tmp_single_batch = input_tmp[:, i : i + 1, :, :]

            output_tmp_single_batch = (
                F.conv2d(
                    input_tmp_single_batch.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, 1, h2, w2),
                    weight=weigth_sq,
                    bias=None,
                    stride=self.stride,
                    padding=self.dw_padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                .reshape(c2, *input_tmp_single_batch.shape[4:], c1, kernel_h, kernel_w)
                .movedim((-3, -2, -1), (1, 2, 3))
            )

            output_tmp_single_batch = torch.flip(output_tmp_single_batch, [-4, -3])
            # reshape as a (num of weights)x(num of column) matrix
            output_tmp_single_batch = output_tmp_single_batch.reshape(c2 * c1 * kernel_h * kernel_w)
            output_tmp[i, :] = output_tmp_single_batch

        if self.bias is not None:
            bias_term = tmp_diag.reshape(b, c2, h2 * w2)
            bias_term = torch.sum(bias_term, 2)
            output_tmp = torch.cat([output_tmp, bias_term], dim=1)

        return output_tmp

    def _jacobian_wrt_weight_sandwich_diag_to_diag_multipoint(
        self, x: Tensor, xB: Tensor, val: Tensor, valB: Tensor, tmp_diag: Tensor
    ) -> Tensor:

        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]
        _, _, kernel_h, kernel_w = self.weight.shape

        input_tmp = tmp_diag.reshape(b, c2, h2, w2)
        # transpose the images in (output height)x(output width)
        input_tmp = torch.flip(input_tmp, [-3, -2, -1])
        # switch batch size and output channel
        input_tmp = input_tmp.movedim(0, 1)

        # define moving sum for Jt_tmp
        output_tmp = torch.zeros(b, c2 * c1 * kernel_h * kernel_w, device=x.device)
        flip_input = torch.flip(x, [-3, -2, -1]).movedim(0, 1)
        flip_inputB = torch.flip(xB, [-3, -2, -1]).movedim(0, 1)
        flip_squared_input = flip_input * flip_inputB

        for i in range(b):
            # set the weight to the convolution
            weigth_sq = flip_squared_input[:, i : i + 1, :, :]
            input_tmp_single_batch = input_tmp[:, i : i + 1, :, :]

            output_tmp_single_batch = (
                F.conv2d(
                    input_tmp_single_batch.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, 1, h2, w2),
                    weight=weigth_sq,
                    bias=None,
                    stride=self.stride,
                    padding=self.dw_padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                .reshape(c2, *input_tmp_single_batch.shape[4:], c1, kernel_h, kernel_w)
                .movedim((-3, -2, -1), (1, 2, 3))
            )

            output_tmp_single_batch = torch.flip(output_tmp_single_batch, [-4, -3])
            # reshape as a (num of weights)x(num of column) matrix
            output_tmp_single_batch = output_tmp_single_batch.reshape(c2 * c1 * kernel_h * kernel_w)
            output_tmp[i, :] = output_tmp_single_batch

        if self.bias is not None:
            bias_term = tmp_diag.reshape(b, c2, h2 * w2)
            bias_term = torch.sum(bias_term, 2)
            output_tmp = torch.cat([output_tmp, bias_term], dim=1)

        return output_tmp
