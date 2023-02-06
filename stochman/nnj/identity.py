import torch
from torch import nn, Tensor

from typing import Optional, Tuple, List, Union

class Identity(nn.Module):
    """Identity module that will return the same input as it receives."""

    def __init__(self):
        super().__init__()

    # def forward(self, x: Tensor, jacobian: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    #     val = x

    #     if jacobian:
    #         xs = x.shape
    #         jac = (
    #             torch.eye(xs[1:].numel(), xs[1:].numel(), dtype=x.dtype, device=x.device)
    #             .repeat(xs[0], 1, 1)
    #             .reshape(xs[0], *xs[1:], *xs[1:])
    #         )
    #         return val, jac
    #     return val

    def _jacobian(
        self, x: Tensor, val: Union[Tensor, None] = None, wrt: str = "input"
    ) -> Union[Tensor, None]:
        if wrt == "input":
            xs = x.shape
            jacobian = (
                torch.eye(xs[1:].numel(), xs[1:].numel(), dtype=x.dtype, device=x.device)
                .repeat(xs[0], 1, 1)
                .reshape(xs[0], *xs[1:], *xs[1:])
            )
            return jacobian

    # def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
    #     return jac_in

    def _jvp(
        self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input"
    ) -> Union[Tensor, None]:
        if wrt == "input":
            return vector

    def _vjp(
        self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input"
    ) -> Union[Tensor, None]:
        if wrt == "input":
            return vector

    def _jmp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Union[Tensor, None]:
        if wrt == "input":
            return matrix

    def _mjp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Union[Tensor, None]:
        if wrt == "input":
            return matrix


def identity(x: Tensor) -> Tensor:
    """Function that for a given input x returns the corresponding identity jacobian matrix"""
    m = Identity()
    return m._jacobian(x)

