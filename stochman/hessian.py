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
        assert loss_func in ("mse", "cross_entropy_binary", "cross_entropy_multiclass", "contrastive_pos", "contrastive_full", "contrastive_fix")
        assert shape in ("full", "block", "diagonal")
        assert speed in ("slow", "half", "fast")

        self.wrt = wrt
        self.loss_func = loss_func
        self.shape = shape
        self.speed = speed
        if speed == "slow":
           # second order
            raise NotImplementedError

    def compute_loss(self, x, target, nnj_module, tuple_indices = None):

        if self.loss_func == "mse":
            return self.compute_mse(x, target, nnj_module)
        elif "cross_entropy" in self.loss_func:
            return self.compute_cross_entropy(x, target, nnj_module)
        elif "contrastive" in self.loss_func:
            return self.compute_contrastive(x, nnj_module, tuple_indices) #contrastive loss does not have targets

    def compute_gradient(self, x, target, nnj_module, tuple_indices = None):

        with torch.no_grad():
            if self.loss_func == "mse":
                return self.compute_mse_gradient(x, target, nnj_module)
            elif "cross_entropy" in self.loss_func:
                raise NotImplementedError
            elif "contrastive" in self.loss_func:
                raise NotImplementedError

    def compute_hessian(self, x, nnj_module, tuple_indices = None):

        with torch.no_grad():
            if self.loss_func == "mse":
                return self.compute_mse_hessian(x, nnj_module)
            elif "cross_entropy" in self.loss_func:
                return self.compute_cross_entropy_hessian(x, nnj_module)
            elif "contrastive" in self.loss_func:
                return self.compute_contrastive_hessian(x, nnj_module, tuple_indices)

    
    def compute_mse(self, x, target, nnj_module):

        val = nnj_module(x)
        assert val.shape == target.shape

        # compute Gaussian log-likelihood
        mse = 0.5*(val - target)**2

        # average along batch size
        mse = torch.mean(mse, dim=0)
        return mse
    
    def compute_mse_gradient(self, x, target, nnj_module):

        val = nnj_module(x)
        assert val.shape == target.shape

        # compute gradient of the Gaussian log-likelihood
        gradient = val - target

        # backpropagate through the network
        gradient = nnj_module._vjp(x, val, gradient, 
                                   wrt=self.wrt),

        # average along batch size
        gradient = torch.mean(gradient, dim=0)
        return gradient

    def compute_mse_hessian(self, x, nnj_module):

        # compute Jacobian sandwich of the identity for each element in the batch
        # H = identity matrix (None is interpreted as identity by jTmjp)

        # backpropagate through the network
        Jt_J = nnj_module._jTmjp(x, None, None,
                                 wrt = self.wrt, 
                                 to_diag = self.shape=="diagonal", 
                                 diag_backprop = self.speed=="fast")
        # average along batch size
        Jt_J = torch.mean(Jt_J, dim=0)
        return Jt_J


    def compute_cross_entropy(self, x, target, nnj_module):

        val = nnj_module(x)
        assert val.shape == target.shape

        if self.loss_func == "cross_entropy_binary":
            bernoulli_p = torch.exp(val) / (1 + torch.exp(val))
            cross_entropy = - (target * torch.log(bernoulli_p) + (1-target) * torch.log(bernoulli_p))

        if self.loss_func == "cross_entropy_multiclass":
            bernoulli_p = ...
            cross_entropy = ...

        # average along batch size
        cross_entropy = torch.mean(cross_entropy, dim=0)
        return cross_entropy

    def compute_cross_entropy_hessian(self, x, nnj_module):

        val = nnj_module(x)

        # initialize the hessian H of the cross entropy
        if self.loss_func == "cross_entropy_binary":
            bernoulli_p = torch.exp(val) / (1 + torch.exp(val))
            H = bernoulli_p - bernoulli_p**2  # hessian in diagonal form

        if self.loss_func == "cross_entropy_multiclass":
            H = ...

        # backpropagate through the network
        Jt_H_J = nnj_module._jTmjp(x, val, H,
                                   wrt = self.wrt, 
                                   from_diag = True,
                                   to_diag = self.shape=="diagonal", 
                                   diag_backprop = self.speed=="fast")
        # average along batch size
        Jt_H_J = torch.mean(Jt_H_J, dim=0)
        return Jt_H_J

    
    def compute_contrastive(self, x, nnj_module, tuple_indices):
        '''
        Notice that the contrastive loss value is the same for 
            self.loss_func == "contrastive_full"
        and
            self.loss_func == "contrastive_fix"
        '''

        # unpack tuple indices
        if len(tuple_indices) == 3:
            a, p, n = tuple_indices
            ap = an = a
        else:
            ap, p, an, n = tuple_indices
        assert len(ap)==len(p) and len(an)==len(n)

        # compute positive part
        pos = 0.5 * (nnj_module(x[ap]) - nnj_module(x[p]))**2

        # sum along batch size
        pos = torch.sum(pos, dim=0)

        if self.loss_func == "contrastive_pos":
            return pos
        
        # compute negative part
        neg = 0.5 * (nnj_module(x[an]) - nnj_module(x[n]))**2

        # sum along batch size
        neg = torch.sum(neg, dim=0)
        
        return pos - neg

    def compute_contrastive_hessian(self, x, nnj_module, tuple_indices):

        # unpack tuple indices
        if len(tuple_indices) == 3:
            a, p, n = tuple_indices
            ap = an = a
        else:
            ap, p, an, n = tuple_indices
        assert len(ap)==len(p) and len(an)==len(n)

        if self.loss_func == "contrastive_full" or self.loss_func == "contrastive_pos":

            # compute positive part
            pos = nnj_module._jTmjp_batch2(x[ap], x[p], None, None, None,
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
            neg = nnj_module._jTmjp_batch2(x[an], x[n], None, None, None,
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
            pos = nnj_module._jTmjp(positives, None, None,
                                    wrt = self.wrt, 
                                    to_diag = self.shape=="diagonal", 
                                    diag_backprop = self.speed=="fast")
            # sum along batch size
            pos = torch.sum(pos, dim=0)

            # compute negative part
            neg = nnj_module._jTmjp(negatives, None, None,
                                    wrt = self.wrt, 
                                    to_diag = self.shape=="diagonal", 
                                    diag_backprop = self.speed=="fast")
            # sum along batch size
            neg = torch.sum(pos, dim=0)

            return pos - neg