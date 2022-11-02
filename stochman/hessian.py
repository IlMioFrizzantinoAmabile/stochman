#!/usr/bin/env python3
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch import nn


class HessianCalculator(ABC, nn.Module):
    def __init__(
        self,
        wrt = "weight",
        loss_func = "mse",
        shape = "diagonal",
        speed = "half",
    ) -> None:
        super().__init__()

        assert wrt in ("weight", "input")
        assert loss_func in ("mse", "cross_entropy", "contrastive_pos", "contrastive_full", "contrastive_fix")
        assert shape in ("full", "block", "diagonal")
        assert speed in ("slow", "half", "fast")

        self.wrt = wrt
        self.loss_func = loss_func
        self.shape = shape
        self.speed = speed
        if speed == "slow":
           # second order
            raise NotImplementedError

    def compute_hessian(self, x, sequential, tuple_indices = None):

        if self.loss_func == "mse":
            return self.compute_mse_hessian(x, sequential)
        elif self.loss_func == "cross_entropy":
            return self.compute_cross_entropy_hessian(x, sequential)
        elif "contrastive" in self.loss_func:
            return self.compute_contrastive_hessian(x, sequential, tuple_indices)

    def compute_mse_hessian(self, x, sequential):

        # compute Jacobian sandwich of the identity for each element in the batch
        Jt_J = sequential._jTmjp(x, None, None,
                                 wrt = self.wrt, 
                                 to_diag = self.shape=="diagonal", 
                                 diag_backprop = self.speed=="fast")
        # average along batch size
        Jt_J = torch.mean(Jt_J, dim=0)
        return Jt_J

    def compute_cross_entropy_hessian(self, x, sequential):

        # init tmp
        tmp = ...

        return ...

    def compute_contrastive_hessian(self, x, sequential, tuple_indices):

        # unpack tuple indices
        if len(tuple_indices) == 3:
            a, p, n = tuple_indices
            ap = an = a
        else:
            ap, p, an, n = tuple_indices
        assert len(ap)==len(p) and len(an)==len(n)

        if self.loss_func == "contrastive_full" or self.loss_func == "contrastive_pos":

            # compute positive part
            pos = sequential._jTmjp_batch2(x[ap], x[p], None, None, None,
                                           wrt = self.wrt,
                                           to_diag = self.shape=="diagonal", 
                                           diag_backprop = self.speed=="fast")
            if self.shape=="diagonal":
                if self.wrt == "weight":
                    pos = [matrixes[0] - 2 * matrixes[1] + matrixes[2] for matrixes in pos]
                    pos = torch.cat(pos, dim=1)
                else:
                    pos = pos[0] - 2 * pos[1] + pos[2]
            else:
                raise NotImplementedError
            # sum along batch size
            pos = torch.sum(pos, dim=0)

            if self.loss_func == "contrastive_pos":
                return pos
            
            # compute negative part
            neg = sequential._jTmjp_batch2(x[an], x[n], None, None, None,
                                           wrt = self.wrt,
                                           to_diag = self.shape=="diagonal", 
                                           diag_backprop = self.speed=="fast")
            if self.shape=="diagonal":
                if self.wrt == "weight":
                    neg = [matrixes[0] - 2 * matrixes[1] + matrixes[2] for matrixes in neg]
                    neg = torch.cat(neg, dim=1)
                else:
                    neg = neg[0] - 2 * neg[1] + neg[2]
            else:
                raise NotImplementedError
            # sum along batch size
            neg = torch.sum(neg, dim=0)
            
            return pos - neg

        if self.loss_func == "contrastive_fix":
            
            positives = x[p] if len(tuple_indices) == 3 else torch.cat((x[ap], x[p]))
            negatives = x[n] if len(tuple_indices) == 3 else torch.cat((x[an], x[n]))

            # compute positive part
            pos = sequential._jTmjp(positives, None, None,
                                    wrt = self.wrt, 
                                    to_diag = self.shape=="diagonal", 
                                    diag_backprop = self.speed=="fast")
            # sum along batch size
            pos = torch.sum(pos, dim=0)

            # compute negative part
            neg = sequential._jTmjp(negatives, None, None,
                                    wrt = self.wrt, 
                                    to_diag = self.shape=="diagonal", 
                                    diag_backprop = self.speed=="fast")
            # sum along batch size
            neg = torch.sum(pos, dim=0)

            return pos - neg
            