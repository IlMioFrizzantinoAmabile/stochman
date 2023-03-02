import torch
from torch import nn, Tensor
from stochman.nnj.abstract_jacobian import AbstractJacobian
from stochman.nnj.identity import identity

from typing import Optional, Tuple, List, Union

class Sequential(AbstractJacobian, nn.Sequential):

    def __init__(self, *args, add_hooks: bool = False):
        super().__init__(*args)
        self._modules_list = list(self._modules.values())

        self.add_hooks = add_hooks
        if self.add_hooks:
            self.feature_maps = []
            self.handles = []
            # def fw_hook(module, input, output):
            #    self.feature_maps.append(output.detach())
            for k in range(len(self._modules)):
                # self.handles.append(self._modules_list[k].register_forward_hook(fw_hook))
                self.handles.append(
                    self._modules_list[k].register_forward_hook(
                        lambda m, i, o: self.feature_maps.append(o.detach())
                    )
                )

    def forward(
        self, x: Tensor, jacobian: Union[Tensor, bool] = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if self.add_hooks:
            self.feature_maps = [x]
        if not (jacobian is False):
            j = identity(x) if (not isinstance(jacobian, Tensor) and jacobian) else jacobian
        for module in self._modules.values():
            val = module(x)
            if not (jacobian is False):
                #print(j.shape)
                # j = module._jacobian_wrt_input_mult_left_vec(x, val, j)
                j = module._jmp(x, val, j)
            x = val
        if not (jacobian is False):
            return x, j
        return x

    def _vjp(self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input") -> Tensor:
        """
        vector jacobian product
        """
        # forward pass for computing hook values
        if val is None:
            val = self.forward(x)

        # backward pass
        vs = []
        for k in range(len(self._modules_list) - 1, -1, -1):
            # backpropagate through the weight
            if wrt == "weight":
                v_k = self._modules_list[k]._vjp(
                    self.feature_maps[k], self.feature_maps[k + 1], vector, wrt="weight"
                )
                if v_k is not None:
                    vs = v_k + vs if isinstance(v_k, list) else [v_k] + vs 
                if k == 0:
                    break
            # backpropagate through the input
            vector = self._modules_list[k]._vjp(
                self.feature_maps[k], self.feature_maps[k + 1], vector, wrt="input"
            )
        if wrt == "weight":
            return torch.cat(vs, dim=1)
        elif wrt == "input":
            return vector


    def _jmp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        jacobian matrix product
        """
        if wrt == "input":
            val, jmp = self.forward(x, jacobian = matrix)
            return jmp
        elif wrt == "weight":
            raise NotImplementedError

    def _mjp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        matrix jacobian product
        """
        # forward pass
        if val is None:
            val = self.forward(x)
        # backward pass
        ms = []
        for k in range(len(self._modules_list) - 1, -1, -1):
            # backpropagate through the weight
            if wrt == "weight":
                m_k = self._modules_list[k]._mjp(
                    self.feature_maps[k], self.feature_maps[k + 1], matrix, wrt="weight"
                )
                if m_k is not None:
                    ms = m_k + ms if isinstance(m_k, list) else [m_k] + ms
                if k == 0:
                    break
            # backpropagate through the input
            matrix = self._modules_list[k]._mjp(
                self.feature_maps[k], self.feature_maps[k + 1], matrix, wrt="input"
            )
        if wrt == "weight":
            return torch.cat(ms, dim=2)
        elif wrt == "input":
            return matrix

    def _jmjTp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, Tuple]:
        """
        jacobian matrix jacobian.T product
        """
        if matrix is None:
            matrix = torch.ones_like(x)
            from_diag = True
        # forward pass
        if val is None:
            val = self.forward(x)
        # forward pass again
        ms = []
        for k in range(len(self._modules_list)):
            # propagate through the weight
            if wrt == "weight":
                raise NotImplementedError
            # propagate through the input
            matrix = self._modules_list[k]._jmjTp(
                self.feature_maps[k],
                self.feature_maps[k + 1],
                matrix,
                wrt="input",
                from_diag=from_diag if k == 0 else diag_backprop,
                to_diag=to_diag if k == len(self._modules_list) - 1 else diag_backprop,
                diag_backprop=diag_backprop,
            )
        if wrt == "input":
            return matrix
        elif wrt == "weight":
            raise NotImplementedError

    def _jTmjp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, Tuple]:
        """
        jacobian.T matrix jacobian product
        """
        # forward pass
        if val is None:
            val = self.forward(x)
        if matrix is None:
            matrix = torch.ones((val.shape[0], val.shape[1:].numel()))
            from_diag = True
        # backward pass
        ms = []
        for k in range(len(self._modules_list) - 1, -1, -1):
            # backpropagate through the weight
            if wrt == "weight":
                m_k = self._modules_list[k]._jTmjp(
                    self.feature_maps[k],
                    self.feature_maps[k + 1],
                    matrix,
                    wrt="weight",
                    from_diag=from_diag if k == len(self._modules_list) - 1 else diag_backprop,
                    to_diag=to_diag,
                    diag_backprop=diag_backprop,
                )
                if m_k is not None:
                    ms = m_k + ms if isinstance(m_k, list) else [m_k] + ms
                if k == 0:
                    break
            # backpropagate through the input
            matrix = self._modules_list[k]._jTmjp(
                self.feature_maps[k],
                self.feature_maps[k + 1],
                matrix,
                wrt="input",
                from_diag=from_diag if k == len(self._modules_list) - 1 else diag_backprop,
                to_diag=to_diag if k == 0 else diag_backprop,
                diag_backprop=diag_backprop,
            )
        if wrt == "input":
            return matrix
        elif wrt == "weight":
            if len(ms) == 0:
                return None  # case of a Sequential with no parametric layers inside
            if to_diag:
                return torch.cat(ms, dim=1)
            else:
                return ms

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
    ):  # -> Union[Tensor, Tuple]:
        """
        jacobian.T matrix jacobian product, computed on 2 inputs (considering cross terms)
        """
        assert x1.shape == x2.shape

        # forward passes
        self.forward(x1)
        feature_maps_1 = self.feature_maps
        self.forward(x2)
        feature_maps_2 = self.feature_maps

        if matrixes is None:
            matrixes = tuple(torch.ones_like(self.feature_maps[-1]) for _ in range(3))
            from_diag = True

        # backward pass
        ms = tuple(([], [], []))
        for k in range(len(self._modules_list) - 1, -1, -1):
            # print('layer:',self._modules_list[k])
            if wrt == "weight":
                m_k = self._modules_list[k]._jTmjp_batch2(
                    feature_maps_1[k],
                    feature_maps_2[k],
                    feature_maps_1[k + 1],
                    feature_maps_2[k + 1],
                    matrixes,
                    wrt="weight",
                    from_diag=from_diag if k == len(self._modules_list) - 1 else diag_backprop,
                    to_diag=to_diag,
                    diag_backprop=diag_backprop,
                )
                if m_k is not None:
                    if all(isinstance(m_k[i], list) for i in range(3)):
                        ms = tuple(m_k[i] + ms[i] for i in range(3) )
                    else:
                        #print(m_k[0].shape)
                        ms = tuple([m_k[i]] + ms[i] for i in range(3) )
                if k == 0:
                    break
            matrixes = self._modules_list[k]._jTmjp_batch2(
                feature_maps_1[k],
                feature_maps_2[k],
                feature_maps_1[k + 1],
                feature_maps_2[k + 1],
                matrixes,
                wrt="input",
                from_diag=from_diag if k == len(self._modules_list) - 1 else diag_backprop,
                to_diag=to_diag if k == 0 else diag_backprop,
                diag_backprop=diag_backprop,
            )
        if wrt == "input":
            return matrixes
        elif wrt == "weight":
            if all(len(ms[i]) for i in range(3)) == 0:
                return None  # case of a Sequential with no parametric layers
            if to_diag:
                return tuple( torch.cat(m, dim=1) for m in ms)
            else:
                return ms