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
        speed = "half"
    ) -> None:
        super().__init__()

        assert wrt in ("weight", "input")
        assert loss_func in ("mse", "cross_entropy", "contrastive")
        assert shape in ("full", "block", "diagonal")
        assert speed in ("slow", "half", "fast")

        self.wrt = wrt
        self.loss_func = loss_func
        self.shape = shape
        if speed == "fast":
            self.method = self.shape + " approx"
        if speed == "half":
            self.method = self.shape + " exact"
        if speed == "slow":
            self.method = self.shape + " second order"
            raise NotImplementedError

    def compute_hessian(self, x, sequential, tuple_indices = None):

        if self.loss_func == "mse":
            return self.compute_mse_hessian(x, sequential)
        elif self.loss_func == "cross_entropy":
            return self.compute_cross_entropy_hessian(x, sequential)
        elif self.loss_func == "contrastive":
            return self.compute_contrastive_hessian(x, sequential, tuple_indices)

    def compute_mse_hessian(self, x, sequential):

        # compute Jacobian sandwich of the identity for each element in the batch
        Jt_J = sequential._jacobian_sandwich(x, None, 
                                            tmp_is_identity = True,
                                            wrt = self.wrt, 
                                            method = self.method)
        # average along batch size
        Jt_J = torch.mean(Jt_J, dim=0)
        return Jt_J

    def compute_cross_entropy_hessian(self, x, sequential):

        # init tmp
        tmp = ...

        return ...

    def compute_contrastive_hessian(self, x, sequential, tuple_indices):
        
        # unpack tuple indices
        ap, p, an, n = tuple_indices
        assert len(ap)==len(p) and len(an)==len(n)

        # compute positive part
        ancor_positives = x[ap]
        positives = x[p]
        pos = sequential._jacobian_sandwich_multipoint(ancor_positives, positives, None,
                                                       tmp_is_identity = True,
                                                       wrt = self.wrt, 
                                                       method = self.method)
        # sum along batch size
        pos = torch.sum(pos, dim=0)
        
        # compute negative part
        ancor_negatives = x[an]
        negatives = x[n]
        neg = sequential._jacobian_sandwich_multipoint(ancor_negatives, negatives, None,
                                                       tmp_is_identity = True,
                                                       wrt = self.wrt, 
                                                       method = self.method)
        # sum along batch size
        neg = torch.sum(neg, dim=0)
        
        return pos - neg