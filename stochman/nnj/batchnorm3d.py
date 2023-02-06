import torch
from torch import nn, Tensor
from stochman.nnj.abstract_jacobian import AbstractJacobian

from typing import Optional, Tuple, List, Union

class BatchNorm3d(AbstractJacobian, nn.BatchNorm3d):

    # only implements jacobian during testing

    def _jacobian(
        self, x: Tensor, val: Union[Tensor, None] = None, wrt: str = "input", diag: bool = True
    ) -> Union[Tensor, None]:
        """Returns the Jacobian matrix"""
        if wrt == "input":
            diag_jacobian = (self.weight / (self.running_var + self.eps).sqrt()).unsqueeze(0)
            if diag:
                return diag_jacobian
            else:
                return torch.diag_embed(diag_jacobian)
        elif wrt == "weight":
            raise NotImplementedError