import torch
import torch.nn.functional as F
from torch import nn, Tensor
from stochman.nnj.abstract_jacobian import AbstractJacobian

from typing import Optional, Tuple, List, Union

class Linear_multihead(AbstractJacobian, nn.Linear):
    def __init__(self, in_features: int, out_heads: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__(in_features, out_features*out_heads, bias, device, dtype)
        self.out_features = out_features
        self.out_heads = out_heads
        self.backprop_single_head = False
    
    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias).reshape(input.shape[0], self.out_heads, self.out_features)

    def _vjp(
        self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input"
    ) -> Tensor:
        """
        vector jacobian product
        """
        if wrt == "input":
            if self.backprop_single_head is False:
                assert vector.shape[1] == self.out_features*self.out_heads
                return torch.einsum("bj,jk->bk", vector, self.weight)
            else:
                assert vector.shape[1] == self.out_features
                in_f = self.weight.shape[1]
                single_head_weight = self.weight.view(self.out_heads, self.out_features, in_f)[self.backprop_single_head].squeeze(0)
                return torch.einsum("bj,jk->bk", vector, single_head_weight )
        elif wrt == "weight":
            b, in_f = x.shape
            out_f = self.out_features
            if self.backprop_single_head is False:
                assert vector.shape[1] == self.out_features*self.out_heads
                if self.bias is None:
                    return torch.einsum("bi,bj->bij", vector, x).view(b, -1)
                else:
                    return torch.cat([torch.einsum("bi,bj->bij", vector, x).view(b, -1), vector], dim=1)
            else:
                assert vector.shape[1] == self.out_features
                h, H = self.backprop_single_head, self.out_heads
                if self.bias is None:
                    vjp = torch.einsum("bi,bj->bij", vector, x).view(b, -1)
                    #vjp = torch.cat([torch.zeros(b, h*out_f*in_f, device=x.device),
                    #                  torch.einsum("bi,bj->bij", vector, x).view(b, -1),
                    #                  torch.zeros(b, (H-h-1)*out_f*in_f, device=x.device)], dim=1)
                    return vjp
                else:
                    vjp = torch.cat([torch.einsum("bi,bj->bij", vector, x).view(b, -1),
                                      vector], dim=1)
                    #vjp = torch.cat([torch.zeros(b, h*out_f*in_f, device=x.device),
                    #                  torch.einsum("bi,bj->bij", vector, x).view(b, -1),
                    #                  torch.zeros(b, (H-h-1)*out_f*in_f + h*out_f, device=x.device),
                    #                  vector,
                    #                  torch.zeros(b, (H-h-1)*out_f, device=x.device)], dim=1)
                    return vjp

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
        if self.backprop_single_head is not False:
            raise NotImplementedError
        if val is None:
            val = self.forward(x)
        if matrix is None:
            matrix = torch.ones_like(val)
            from_diag = True
        if wrt == "input":
            if not from_diag and not to_diag:
                # full -> full
                return torch.einsum("nm,bnj,jk->bmk", self.weight, matrix, self.weight)
            elif from_diag and not to_diag:
                # diag -> full
                return torch.einsum("nm,bn,nk->bmk", self.weight, matrix, self.weight)
            elif not from_diag and to_diag:
                # full -> diag
                return torch.einsum("nm,bnj,jm->bm", self.weight, matrix, self.weight)
            elif from_diag and to_diag:
                # diag -> diag
                return torch.einsum("nm,bn,nm->bm", self.weight, matrix, self.weight)
        elif wrt == "weight":
            if not from_diag and not to_diag:
                # full -> full
                #TODO
                jacobian = self._jacobian_wrt_weight(x, val)
                return torch.einsum("bji,bjk,bkq->biq", jacobian, matrix, jacobian)
            elif from_diag and not to_diag:
                # diag -> full
                #TODO
                jacobian = self._jacobian_wrt_weight(x, val)
                return torch.einsum("bji,bj,bjq->biq", jacobian, matrix, jacobian)
            elif not from_diag and to_diag:
                # full -> diag
                bs, _, _ = matrix.shape
                x_sq = x * x
                if self.bias is None:
                    return torch.einsum("bj,bii->bij", x_sq, matrix).view(bs, -1)
                else:
                    return torch.cat(
                        [
                            torch.einsum("bj,bii->bij", x_sq, matrix).view(bs, -1),
                            torch.einsum("bii->bi", matrix),
                        ],
                        dim=1,
                    )
            elif from_diag and to_diag:
                # diag -> diag
                bs, _ = matrix.shape
                x_sq = x * x
                if self.bias is None:
                    return torch.einsum("bj,bi->bij", x_sq, matrix).view(bs, -1)
                else:
                    return torch.cat([torch.einsum("bj,bi->bij", x_sq, matrix).view(bs, -1), matrix], dim=1)
