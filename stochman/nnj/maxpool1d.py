import torch
import torch.nn.functional as F
from torch import nn, Tensor
from stochman.nnj.abstract_jacobian import AbstractJacobian

from typing import Optional, Tuple, List, Union

class MaxPool1d(AbstractJacobian, nn.MaxPool1d):
    def forward(self, input: Tensor):
        val, idx = F.max_pool1d(
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

    def _jmp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        jacobian matrix product
        """
        if wrt == "input":
            if val is None:
                val = self.forward(x)
            b, c1, l1 = x.shape
            c2, l2 = val.shape[1:]

            matrix_orig_shape = matrix.shape
            matrix = matrix.reshape(-1, l1, *matrix_orig_shape[3:])
            arange_repeated = torch.repeat_interleave(torch.arange(b * c1), l2).long()
            idx = self.idx.reshape(-1)
            matrix = matrix[arange_repeated, idx, :, :].reshape(*val.shape, *matrix_orig_shape[3:])
            return matrix
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None