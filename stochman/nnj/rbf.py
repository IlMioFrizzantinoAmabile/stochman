import torch
from torch import nn, Tensor
from stochman.nnj.abstract_jacobian import AbstractJacobian

from typing import Optional, Tuple, List, Union

class RBF(AbstractJacobian, nn.Module):
    def __init__(self, dim, num_points, points=None, beta=1.0):
        super().__init__()
        if points is None:
            self.points = nn.Parameter(torch.randn(num_points, dim))
        else:
            self.points = nn.Parameter(points, requires_grad=False)
        if isinstance(beta, torch.Tensor):
            self.beta = beta.view(1, -1)
        else:
            self.beta = beta

    def __dist2__(self, x):
        x_norm = (x**2).sum(1).view(-1, 1)
        points_norm = (self.points**2).sum(1).view(1, -1)
        d2 = x_norm + points_norm - 2.0 * torch.mm(x, self.points.transpose(0, 1))
        return d2.clamp(min=0.0)  # NxM
        # if x.dim() is 2:
        #    x = x.unsqueeze(0) # BxNxD
        # x_norm = (x**2).sum(-1, keepdim=True) # BxNx1
        # points_norm = (self.points**2).sum(-1, keepdim=True).view(1, 1, -1) # 1x1xM
        # d2 = x_norm + points_norm - 2.0 * torch.bmm(x, self.points.t().unsqueeze(0).expand(x.shape[0], -1, -1))
        # return d2.clamp(min=0.0) # BxNxM

    def forward(self, x, jacobian=False):
        D2 = self.__dist2__(x)  # (batch)-by-|x|-by-|points|
        val = torch.exp(-self.beta * D2)  # (batch)-by-|x|-by-|points|

        if jacobian:
            J = self._jacobian(x, val)
            return val, J
        else:
            return val

    def _jacobian(
        self, x: Tensor, val: Union[Tensor, None] = None, wrt: str = "input", diag: bool = False
    ) -> Union[Tensor, None]:
        """Returns the Jacobian matrix"""
        if wrt == "input":
            if diag==True:
                raise NotImplementedError
            if val is None:
                val = self.forward(x)
            T1 = -2.0 * self.beta * val  # BxNxM
            T2 = x.unsqueeze(1) - self.points.unsqueeze(0)
            jacobian = T1.unsqueeze(-1) * T2
            return jacobian
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
            return torch.einsum("bij,bjk->bik", jacobian, matrix)
        elif wrt == "weight":
            return None