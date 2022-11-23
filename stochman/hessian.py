#!/usr/bin/env python3
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch import nn


class HessianCalculator(ABC, nn.Module):
    def __init__(
        self,
        wrt="weight",
        loss_func="mse",
        shape="diagonal",
        speed="half",
    ) -> None:
        super().__init__()

        assert wrt in ("weight", "input")
        assert loss_func in (
            "mse",
            "cross_entropy_binary",
            "cross_entropy_multiclass",
            "contrastive_full",
            "contrastive_pos",
            "contrastive_fix",
            "arccos_full",
            "arccos_pos",
        )
        assert shape in ("full", "block", "diagonal")
        assert speed in ("slow", "half", "fast")

        self.wrt = wrt
        self.loss_func = loss_func
        self.shape = shape
        self.speed = speed
        if speed == "slow":
            # second order
            raise NotImplementedError

    def compute_loss(self, x, target, nnj_module, tuple_indices=None):
        raise NotImplementedError

    def compute_gradient(self, x, target, nnj_module, tuple_indices=None):
        raise NotImplementedError

    def compute_hessian(self, x, nnj_module, tuple_indices=None):
        raise NotImplementedError


class MSEHessianCalculator(HessianCalculator):
    def compute_loss(self, x, target, nnj_module, tuple_indices=None):

        val = nnj_module(x)
        assert val.shape == target.shape

        # compute Gaussian log-likelihood
        mse = 0.5 * (val - target) ** 2

        # average along batch size
        mse = torch.mean(mse, dim=0)
        return mse

    def compute_gradient(self, x, target, nnj_module, tuple_indices=None):

        val = nnj_module(x)
        assert val.shape == target.shape

        # compute gradient of the Gaussian log-likelihood
        gradient = val - target

        # backpropagate through the network
        gradient = (nnj_module._vjp(x, val, gradient, wrt=self.wrt),)

        # average along batch size
        gradient = torch.mean(gradient, dim=0)
        return gradient

    def compute_hessian(self, x, nnj_module, tuple_indices=None):

        # compute Jacobian sandwich of the identity for each element in the batch
        # H = identity matrix (None is interpreted as identity by jTmjp)

        # backpropagate through the network
        Jt_J = nnj_module._jTmjp(
            x, None, None, wrt=self.wrt, to_diag=self.shape == "diagonal", diag_backprop=self.speed == "fast"
        )
        # average along batch size
        Jt_J = torch.mean(Jt_J, dim=0)
        return Jt_J


class BCEHessianCalculator(HessianCalculator):
    def compute_loss(self, x, target, nnj_module, tuple_indices=None):

        val = nnj_module(x)
        assert val.shape == target.shape

        bernoulli_p = torch.exp(val) / (1 + torch.exp(val))
        cross_entropy = -(target * torch.log(bernoulli_p) + (1 - target) * torch.log(bernoulli_p))

        # average along batch size
        cross_entropy = torch.mean(cross_entropy, dim=0)
        return cross_entropy

    def compute_gradient(self, x, target, nnj_module, tuple_indices=None):
        raise NotImplementedError

    def compute_hessian(self, x, nnj_module, tuple_indices=None):

        val = nnj_module(x)

        bernoulli_p = torch.exp(val) / (1 + torch.exp(val))
        H = bernoulli_p - bernoulli_p**2  # hessian in diagonal form

        # backpropagate through the network
        Jt_H_J = nnj_module._jTmjp(
            x,
            val,
            H,
            wrt=self.wrt,
            from_diag=True,
            to_diag=self.shape == "diagonal",
            diag_backprop=self.speed == "fast",
        )
        # average along batch size
        Jt_H_J = torch.mean(Jt_H_J, dim=0)
        return Jt_H_J


class ContrastiveHessianCalculator(HessianCalculator):
    def compute_loss(self, x, target, nnj_module, tuple_indices):

        """
        L(x,y) = 0.5 * || x - y ||
        Contrastive(x, tuples) = sum_positives L(x,y) - sum_negatives L(x,y)

        Notice that the contrastive loss value is the same for
            self.loss_func == "contrastive_full"
        and
            self.loss_func == "contrastive_fix"
        """

        # unpack tuple indices
        if len(tuple_indices) == 3:
            a, p, n = tuple_indices
            ap = an = a
        else:
            ap, p, an, n = tuple_indices
        assert len(ap) == len(p) and len(an) == len(n)

        # compute positive part
        pos = 0.5 * (nnj_module(x[ap]) - nnj_module(x[p])) ** 2

        # sum along batch size
        pos = torch.sum(pos, dim=0)

        if self.loss_func == "contrastive_pos":
            return pos

        # compute negative part
        neg = 0.5 * (nnj_module(x[an]) - nnj_module(x[n])) ** 2

        # sum along batch size
        neg = torch.sum(neg, dim=0)

        return pos - neg

    def compute_gradient(self, x, target, nnj_module, tuple_indices):
        raise NotImplementedError

    def compute_hessian(self, x, nnj_module, tuple_indices):

        # unpack tuple indices
        if len(tuple_indices) == 3:
            a, p, n = tuple_indices
            ap = an = a
        else:
            ap, p, an, n = tuple_indices
        assert len(ap) == len(p) and len(an) == len(n)

        if self.loss_func == "contrastive_full" or self.loss_func == "contrastive_pos":

            # compute positive part
            pos = nnj_module._jTmjp_batch2(
                x[ap],
                x[p],
                None,
                None,
                None,
                wrt=self.wrt,
                to_diag=self.shape == "diagonal",
                diag_backprop=self.speed == "fast",
            )
            if self.shape == "diagonal":
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
            neg = nnj_module._jTmjp_batch2(
                x[an],
                x[n],
                None,
                None,
                None,
                wrt=self.wrt,
                to_diag=self.shape == "diagonal",
                diag_backprop=self.speed == "fast",
            )
            if self.shape == "diagonal":
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
            pos = nnj_module._jTmjp(
                positives,
                None,
                None,
                wrt=self.wrt,
                to_diag=self.shape == "diagonal",
                diag_backprop=self.speed == "fast",
            )
            # sum along batch size
            pos = torch.sum(pos, dim=0)

            # compute negative part
            neg = nnj_module._jTmjp(
                negatives,
                None,
                None,
                wrt=self.wrt,
                to_diag=self.shape == "diagonal",
                diag_backprop=self.speed == "fast",
            )
            # sum along batch size
            neg = torch.sum(pos, dim=0)

            return pos - neg


class ArccosHessianCalculator(HessianCalculator):
    def compute_loss(self, x, nnj_module, tuple_indices):
        """
        L(x,y) = 0.5 * sum_i x_i * y_i
               = 0.5 * || x / ||x|| - y / ||y|| || - 1    # arccos distance is equivalent to contrastive distance & normalization layer
        Arcos(x, tuples) = sum_positives L(x,y) - sum_negatives L(x,y)
        """

        def _arccos(x1, x2):
            z1 = nnj_module(x1)
            z2 = nnj_module(x2)
            z1_norm = torch.sum(z1**2, dim=1) ** (0.5)
            z2_norm = torch.sum(z2**2, dim=1) ** (0.5)
            return 0.5 * torch.einsum("bi,bi->b", z1, z2) / (z1_norm * z2_norm)

        # unpack tuple indices
        if len(tuple_indices) == 3:
            a, p, n = tuple_indices
            ap = an = a
        else:
            ap, p, an, n = tuple_indices
        assert len(ap) == len(p) and len(an) == len(n)

        # compute positive part
        pos = _arccos(x[ap], x[p])

        # sum along batch size
        pos = torch.sum(pos, dim=0)

        if self.loss_func == "arccos_pos":
            return pos

        # compute negative part
        neg = _arccos(x[an], x[n])

        # sum along batch size
        neg = torch.sum(neg, dim=0)

        return pos - neg

    def compute_gradient(self, x, target, nnj_module, tuple_indices):
        raise NotImplementedError

    def compute_hessian(self, x, nnj_module, tuple_indices):
        def _arccos_hessian(z1, z2):
            z1_norm = torch.sum(z1**2, dim=1) ** (0.5)
            z2_norm = torch.sum(z2**2, dim=1) ** (0.5)
            z1_normalized = torch.einsum("bi,b->bi", z1, 1 / z1_norm)
            z2_normalized = torch.einsum("bi,b->bi", z2, 1 / z2_norm)
            cosine = torch.einsum("bi,bi->b", z1_normalized, z2_normalized)

            zs = z1.shape
            identity = torch.eye(zs[1], dtype=z1.dtype, device=z1.device).repeat(zs[0], 1, 1)
            cosine_times_identity = torch.einsum("bij,b->bij", identity, cosine)

            outer_11 = torch.einsum("bi,bj->bij", z1_normalized, z1_normalized)
            outer_12 = torch.einsum("bi,bj->bij", z1_normalized, z2_normalized)
            outer_21 = torch.einsum("bi,bj->bij", z2_normalized, z1_normalized)
            outer_22 = torch.einsum("bi,bj->bij", z2_normalized, z2_normalized)

            cosine_times_outer_11 = torch.einsum("bij,b->bij", outer_11, cosine)
            cosine_times_outer_21 = torch.einsum("bij,b->bij", outer_21, cosine)
            cosine_times_outer_22 = torch.einsum("bij,b->bij", outer_22, cosine)

            H_11 = cosine_times_identity + outer_12 + outer_21 - 3 * cosine_times_outer_11
            H_12 = -identity + outer_11 + outer_22 - cosine_times_outer_21
            H_22 = cosine_times_identity + outer_12 + outer_21 - 3 * cosine_times_outer_22

            H_11_normalized = torch.einsum("bij,b->bij", H_11, 1 / (z1_norm**2))
            H_12_normalized = torch.einsum("bij,b,b->bij", H_12, 1 / (z1_norm * z2_norm))
            H_22_normalized = torch.einsum("bij,b->bij", H_22, 1 / (z2_norm**2))
            return tuple((H_11_normalized, H_12_normalized, H_22_normalized))

        # unpack tuple indices
        if len(tuple_indices) == 3:
            a, p, n = tuple_indices
            ap = an = a
        else:
            ap, p, an, n = tuple_indices
        assert len(ap) == len(p) and len(an) == len(n)

        if self.loss_func == "arccos_full" or self.loss_func == "arccos_pos":

            # compute positive part

            # forward pass
            z1, z2 = nnj_module(x[ap]), nnj_module(x[p])

            # initialize the hessian of the loss
            H = _arccos_hessian(z1, z2)

            # backpropagate through the network
            pos = nnj_module._jTmjp_batch2(
                x[ap],
                x[p],
                None,
                None,
                H,
                wrt=self.wrt,
                to_diag=self.shape == "diagonal",
                from_diag=False,
                diag_backprop=self.speed == "fast",
            )
            if self.shape == "diagonal":
                if self.wrt == "weight":
                    pos = [matrixes[0] - 2 * matrixes[1] + matrixes[2] for matrixes in pos]
                    pos = torch.cat(pos, dim=1)
                else:
                    pos = pos[0] - 2 * pos[1] + pos[2]
            else:
                raise NotImplementedError
            # sum along batch size
            pos = torch.sum(pos, dim=0)

            if self.loss_func == "arccos_pos":
                return pos

            # compute negative part

            # forward pass
            z1, z2 = nnj_module(x[an]), nnj_module(x[n])

            # initialize the hessian of the loss
            H = _arccos_hessian(z1, z2)

            # backpropagate through the network
            neg = nnj_module._jTmjp_batch2(
                x[an],
                x[n],
                None,
                None,
                H,
                wrt=self.wrt,
                to_diag=self.shape == "diagonal",
                from_diag=False,
                diag_backprop=self.speed == "fast",
            )
            if self.shape == "diagonal":
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
