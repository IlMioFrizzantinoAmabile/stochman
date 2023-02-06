import torch
import torch.nn.functional as F
from torch import nn, Tensor
from stochman.nnj.identity import identity

from typing import Optional, Tuple, List, Union

class AbstractJacobian:
    """Abstract class that:
    - will overwrite the default behaviour of the forward method such that it
    is also possible to return the jacobian
    - propagate jacobian vector and jacobian matrix products, both forward and backward
    - pull back and push forward metrics
    """

    def __call__(self, x: Tensor, jacobian: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        val = self._call_impl(x)
        if jacobian:
            jac = self._jacobian(x, val)
            return val, jac
        return val

    def _jacobian(
        self, x: Tensor, val: Union[Tensor, None] = None, wrt: str = "input", diag: bool = False
    ) -> Union[Tensor, None]:
        """Returns the Jacobian matrix"""
        # this function has to be implemented for every new nnj layer
        raise NotImplementedError
        
    ###############################
    ### jacobians outer product ###
    ###############################

    def jjT(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        wrt: str = "input",
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, None]:
        """ jacobian * jacobian.T """
        return self.jmjTp(
            x, val, None, wrt=wrt, to_diag=to_diag, diag_backprop=diag_backprop
        )

    def jTj(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        wrt: str = "input",
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, List[Tensor], None]:
        """ jacobian.T * jacobian """
        return self.jTmjp(
            x, val, None, wrt=wrt, to_diag=to_diag, diag_backprop=diag_backprop
        )

    ######################
    ### forward passes ###
    ######################

    def jvp(
        self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input"
    ) -> Union[Tensor, None]:
        """jacobian vector product - forward"""
        if wrt == "weight":
            raise NotImplementedError
        if val is None:
            val = self.forward(x)
        xs, vs = x.shape, val.shape
        if vector.shape == xs:
            vector = vector.reshape(xs[0], xs[1:].numel())
            reshape = True
        elif len(vector.shape)==2:
            assert vector.shape[0]==xs[0] and vector.shape[1]==xs[1:].numel()
            reshape = False
        else:
            raise ValueError(f"Invalid vector shape! I need a 2D tensor of shape [{xs[0]}, {xs[1:].numel()}] or {xs}")
        jacobian_vector_product = self._jvp(x, val, vector, wrt=wrt)
        if reshape:
            return jacobian_vector_product.reshape(vs[0], *vs[1:])
        else:
            return jacobian_vector_product

    def jmp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Union[Tensor, None]:
        """jacobian matrix product - forward"""
        if wrt == "weight":
            raise NotImplementedError
        if val is None:
            val = self.forward(x)
        xs, vs = x.shape, val.shape
        if matrix is None:
            matrix = identity(x)
            matrix = matrix.reshape(xs[0], xs[1:].numel(), xs[1:].numel())
        if len(matrix.shape)!=3 or matrix.shape[0]!= xs[0] or matrix.shape[1]!=xs[1:].numel():
            raise ValueError(f"Invalid matrix shape! I need a 3D tensor of shape [{xs[0]}, {xs[1:].numel()}, _]")
        return self._jmp(x, val, matrix, wrt=wrt)

    def jmjTp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, None]:
        """jacobian matrix jacobian.T product - forward"""
        if wrt == "weight":
            raise NotImplementedError
        if val is None:
            val = self.forward(x)
        xs, vs = x.shape, val.shape
        if matrix is None:
            matrix = torch.ones_like(x).reshape(xs[0], xs[1:].numel())
            from_diag = True
        return self._jmjTp(
            x, val, matrix, wrt=wrt, from_diag=from_diag, to_diag=to_diag, diag_backprop=diag_backprop
        )
    

    #######################
    ### backward passes ###
    #######################

    def vjp(
        self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input"
    ) -> Union[Tensor, None]:
        """vector jacobian product - backward"""
        if val is None:
            val = self.forward(x)
        xs, vs = x.shape, val.shape
        if vector.shape == vs:
            vector = vector.reshape(vs[0], vs[1:].numel())
            reshape = True
        elif len(vector.shape)==2:
            assert vector.shape[0]==vs[0] and vector.shape[1]==vs[1:].numel()
            reshape = False
        else:
            raise ValueError(f"Invalid vector shape! I need a 2D tensor of shape [{vs[0]}, {vs[1:].numel()}] or {vs}")
        vector_jacobian_product = self._vjp(x, val, vector, wrt=wrt)
        if wrt == "input":
            if reshape:
                return vector_jacobian_product.reshape(xs[0], *xs[1:])
            else:
                return vector_jacobian_product
        elif wrt == "weight":
            return vector_jacobian_product

    def mjp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Union[Tensor, None]:
        """matrix jacobian product - backward"""
        if val is None:
            val = self.forward(x)
        xs, vs = x.shape, val.shape
        if matrix is None:
            matrix = identity(val)
            matrix = matrix.reshape(vs[0], vs[1:].numel(), vs[1:].numel())
        if len(matrix.shape)!=3 or matrix.shape[0]!= vs[0] or matrix.shape[2]!=vs[1:].numel():
            raise ValueError(f"Invalid matrix shape! I need a 3D tensor of shape [{vs[0]}, _, {vs[1:].numel()}]")
        return self._mjp(x, val, matrix, wrt=wrt)

    def jTmjp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, List[Tensor], None]:
        """jacobian.T matrix jacobian product - backward"""
        if val is None:
            val = self.forward(x)
        xs, vs = x.shape, val.shape
        if matrix is None:
            matrix = torch.ones_like(val).reshape(vs[0], vs[1:].numel())
            from_diag = True
        return self._jTmjp(
            x, val, matrix, wrt=wrt, from_diag=from_diag, to_diag=to_diag, diag_backprop=diag_backprop
        )

    def jTmjp_batch2(
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
        """jacobian.T matrix jacobian product - backward
         computed on 2 inputs (considering cross terms)"""
        assert x1.shape == x2.shape
        if val1 is None:
            val1 = self.forward(x1)
        if val2 is None:
            val2 = self.forward(x2)
        xs, vs = x1.shape, val1.shape
        if matrixes is None:
            matrixes = tuple(torch.ones_like(val1).reshape(vs[0], vs[1:].numel()) for _ in range(3))
            from_diag = True
        matrixes = tuple(matrix.reshape(vs[0], vs[1:].numel(), vs[1:].numel()) for matrix in matrixes)
        jacobianT_matrixes_jacobian_product = self._jTmjp_batch2(
            x1, x2, val1, val2, matrixes, wrt=wrt, from_diag=from_diag, to_diag=to_diag, diag_backprop=diag_backprop
        )
        jacobianT_matrixes_jacobian_product

    ########################################################################################
    ### slow implementations, to be overwritten by each module for efficient computation ###
    ########################################################################################

    def _jvp(self, x: Tensor, val: Tensor, vector: Tensor, wrt: str = "input") -> Union[Tensor, None]:
        print(f"Ei! I ({self}) am doing jvp in the stupid way!")
        jacobian = self._jacobian(x, val, wrt=wrt)
        if jacobian is None: #non parametric layer
            return None
        return torch.einsum("bij,bj->bi", jacobian, vector)

    def _jmp(self, x: Tensor, val: Tensor, matrix: Tensor, wrt: str = "input") -> Union[Tensor, None]:
        print(f"Ei! I ({self}) am doing jmp in the stupid way!")
        jacobian = self._jacobian(x, val, wrt=wrt)
        if jacobian is None: #non parametric layer
            return None
        return torch.einsum("bij,bjk->bik", jacobian, matrix)

    def _jmjTp(
        self,
        x: Tensor,
        val: Tensor,
        matrix: Tensor,
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, None]:
        print(f"Ei! I ({self}) am doing jmjTp in the stupid way!")
        if diag_backprop: #TODO
            raise NotImplementedError
        jacobian = self._jacobian(x, val, wrt=wrt)
        if jacobian is None: #non parametric layer
            return None
        if not from_diag and not to_diag:
            # full -> full
            return torch.einsum("bij,bjk,blk->bil", jacobian, matrix, jacobian)
        elif from_diag and not to_diag:
            # diag -> full
            return torch.einsum("bij,bj,blj->bil", jacobian, matrix, jacobian)
        elif not from_diag and to_diag:
            # full -> diag
            return torch.einsum("bij,bjk,bik->bi", jacobian, matrix, jacobian)
        elif from_diag and to_diag:
            # diag -> diag
            return torch.einsum("bij,bj,bij->bi", jacobian, matrix, jacobian)

    def _vjp(self, x: Tensor, val: Tensor, vector: Tensor, wrt: str = "input") -> Union[Tensor, None]:
        print(f"Ei! I ({self}) am doing vjp in the stupid way!")
        jacobian = self._jacobian(x, val, wrt=wrt)
        if jacobian is None: #non parametric layer
            return None
        return torch.einsum("bi,bij->bj", vector, jacobian)

    def _mjp(self, x: Tensor, val: Tensor, matrix: Tensor, wrt: str = "input") -> Union[Tensor, None]:
        print(f"Ei! I ({self}) am doing mjp in the stupid way!")
        jacobian = self._jacobian(x, val, wrt=wrt)
        if jacobian is None: #non parametric layer
            return None
        return torch.einsum("bij,bjk->bik", matrix, jacobian)

    def _jTmjp(
        self,
        x: Tensor,
        val: Tensor,
        matrix: Tensor,
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, List[Tensor], None]:
        print(f"Ei! I ({self}) am doing jTmjp in the stupid way!")
        if diag_backprop:
            raise NotImplementedError
        jacobian = self._jacobian(x, val, wrt=wrt)
        if jacobian is None: #non parametric layer
            return None
        if not from_diag and not to_diag:
            # full -> full
            return torch.einsum("bji,bjk,bkl->bil", jacobian, matrix, jacobian)
        elif from_diag and not to_diag:
            # diag -> full
            return torch.einsum("bij,bj,bjl->bil", jacobian, matrix, jacobian)
        elif not from_diag and to_diag:
            # full -> diag
            return torch.einsum("bij,bjk,bki->bi", jacobian, matrix, jacobian)
        elif from_diag and to_diag:
            # diag -> diag
            return torch.einsum("bij,bj,bji->bi", jacobian, matrix, jacobian)

    def _jTmjp_batch2(
        self,
        x1: Tensor,
        x2: Tensor,
        val1: Tensor,
        val2: Tensor,
        matrixes: Tuple[Tensor, Tensor, Tensor],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ):  # -> Union[Tensor, Tuple]:
        print(f"Ei! I ({self}) am doing jmjTp_batch2 in the stupid way!")
        if diag_backprop:
            raise NotImplementedError
        j1 = self._jacobian(x1, val1, wrt=wrt)
        j2 = self._jacobian(x2, val2, wrt=wrt)
        if j1 is None or j2 is None: #non parametric layer
            return None
        jTmjps = []
        for j_left, matrix, j_right in ((j1, matrixes[0], j1), (j1, matrixes[1], j2), (j2, matrixes[2], j2)):
            if not from_diag and not to_diag:
                # full -> full
                jTmjps.append( torch.einsum("bji,bjk,bkl->bil", j_left, matrix, j_right) )
            elif from_diag and not to_diag:
                # diag -> full
                jTmjps.append( torch.einsum("bij,bj,bjl->bil", j_left, matrix, j_right) )
            elif not from_diag and to_diag:
                # full -> diag
                jTmjps.append( torch.einsum("bij,bjk,bki->bi", j_left, matrix, j_right) )
            elif from_diag and to_diag:
                # diag -> diag
                jTmjps.append( torch.einsum("bij,bj,bji->bi", j_left, matrix, j_right) )
        return tuple(jTmjps)



class AbstractDiagonalJacobian(AbstractJacobian):
    """
    Superclass specific for layers whose Jacobian is a diagonal matrix.
    In these cases the forward and backward functions can be efficiently implemented in a general form"""


    def _jvp(
        self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input"
    ) -> Union[Tensor, None]:
        """
        jacobian vector product
        """
        if wrt == "input":
            diag_jacobian = self._jacobian(x, val, diag=True)
            return torch.einsum("bj,bj->bj", diag_jacobian, vector)
        elif wrt == "weight":
            return None

    def _jmp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Union[Tensor, None]:
        """
        jacobian matrix product
        """
        if wrt == "input":
            if matrix is None:
                return self._jacobian(x, val, diag=False)
            diag_jacobian = self._jacobian(x, val, diag=True)
            return torch.einsum("bi,bij->bij", diag_jacobian, matrix)
        elif wrt == "weight":
            return None
    
    def _jmjTp(
        self,
        x: Tensor,
        val: Tensor,
        matrix: Tensor,
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, None]:
        """
        jacobian matrix jacobian.T product
        """
        b = x.shape[0]
        if val is None:
            val = self.forward(x)
        if matrix is None:
            matrix = torch.ones_like(x).reshape(b, -1)
            from_diag = True
        if wrt == "input":
            if not from_diag and not to_diag:
                # full -> full
                diag_jacobian = self._jacobian(x, val, diag=True)
                return torch.einsum("bi,bik,bk->bik", diag_jacobian, matrix, diag_jacobian)
            elif from_diag and not to_diag:
                # diag -> full
                diag_jacobian_square = self._jacobian(x, val, diag=True) ** 2
                return torch.diag_embed(torch.einsum("bi,bi->bi", diag_jacobian_square, matrix))
            elif not from_diag and to_diag:
                # full -> diag
                diag_jacobian = self._jacobian(x, val, diag=True)
                return torch.einsum("bi,bii,bi->bi", diag_jacobian, matrix, diag_jacobian)
            elif from_diag and to_diag:
                # diag -> diag
                diag_jacobian_square = self._jacobian(x, val, diag=True) ** 2
                return torch.einsum("bi,bi->bi", diag_jacobian_square, matrix)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

    def _vjp(
        self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input"
    ) -> Union[Tensor, None]:
        """
        vector jacobian product
        """
        if wrt == "input":
            diag_jacobian = self._jacobian(x, val, diag=True)
            return torch.einsum("bi,bi->bi", vector, diag_jacobian)
        elif wrt == "weight":
            return None

    def _mjp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Union[Tensor, None]:
        """
        matrix jacobian product
        """
        if wrt == "input":
            if matrix is None:
                return self._jacobian(x, val, diag=False)
            diag_jacobian = self._jacobian(x, val, diag=True)
            return torch.einsum("bij,bj->bij", matrix, diag_jacobian)
        elif wrt == "weight":
            return None

    def _jTmjp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, None]:
        """
        jacobian.T matrix jacobian product
        """
        b = x.shape[0]
        if val is None:
            val = self.forward(x)
        if matrix is None:
            matrix = torch.ones_like(val).reshape(b, -1)
            from_diag = True
        if wrt == "input":
            if not from_diag and not to_diag:
                # full -> full
                diag_jacobian = self._jacobian(x, val, diag=True)
                return torch.einsum("bi,bik,bk->bik", diag_jacobian, matrix, diag_jacobian)
            elif from_diag and not to_diag:
                # diag -> full
                diag_jacobian_square = self._jacobian(x, val, diag=True) ** 2
                return torch.diag_embed(torch.einsum("bi,bi->bi", diag_jacobian_square, matrix))
            elif not from_diag and to_diag:
                # full -> diag
                diag_jacobian = self._jacobian(x, val, diag=True)
                return torch.einsum("bi,bii,bi->bi", diag_jacobian, matrix, diag_jacobian)
            elif from_diag and to_diag:
                # diag -> diag
                diag_jacobian_square = self._jacobian(x, val, diag=True) ** 2
                return torch.einsum("bi,bi->bi", diag_jacobian_square, matrix)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

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
    ) -> Union[Tuple[Tensor, Tensor, Tensor], None]:
        """
        jacobian.T matrix jacobian product, computed on 2 inputs (considering cross terms)
        """
        assert x1.shape == x2.shape
        b = x1.shape[0]
        if val1 is None:
            val1 = self.forward(x1)
        if val2 is None:
            val2 = self.forward(x2)
        assert val1.shape == val2.shape
        if matrixes is None:
            matrixes = tuple(torch.ones_like(val1).reshape(b, -1) for _ in range(3))
            from_diag = True

        if wrt == "input":
            m11, m12, m22 = matrixes
            jac_1_diag = self._jacobian(x1, val1, diag=True)
            jac_2_diag = self._jacobian(x2, val2, diag=True)

            if not from_diag and not to_diag:
                # full -> full
                return tuple(
                    torch.einsum("bi,bij,bj->bij", jac_i, m, jac_j)
                    for jac_i, m, jac_j in [
                        (jac_1_diag, m11, jac_1_diag),
                        (jac_1_diag, m12, jac_2_diag),
                        (jac_2_diag, m22, jac_2_diag),
                    ]
                )
            elif from_diag and not to_diag:
                # diag -> full
                return tuple(
                    torch.diag_embed(torch.einsum("bi,bi,bi->bi", jac_i, m, jac_j))
                    for jac_i, m, jac_j in [
                        (jac_1_diag, m11, jac_1_diag),
                        (jac_1_diag, m12, jac_2_diag),
                        (jac_2_diag, m22, jac_2_diag),
                    ]
                )
            elif not from_diag and to_diag:
                # full -> diag
                return tuple(
                    torch.einsum("bi,bii,bi->bi", jac_i, m, jac_j)
                    for jac_i, m, jac_j in [
                        (jac_1_diag, m11, jac_1_diag),
                        (jac_1_diag, m12, jac_2_diag),
                        (jac_2_diag, m22, jac_2_diag),
                    ]
                )
            elif from_diag and to_diag:
                # diag -> diag
                return tuple(
                    torch.einsum("bi,bi,bi->bi", jac_i, m, jac_j)
                    for jac_i, m, jac_j in [
                        (jac_1_diag, m11, jac_1_diag),
                        (jac_1_diag, m12, jac_2_diag),
                        (jac_2_diag, m22, jac_2_diag),
                    ]
                )
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None