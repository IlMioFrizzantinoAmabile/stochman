from stochman.nnj.identity import Identity
from stochman.nnj.abstract_jacobian import AbstractJacobian, AbstractDiagonalJacobian

############################
### multi-layer wrappers ###
from stochman.nnj.sequential import Sequential
from stochman.nnj.residualblock import ResidualBlock
from stochman.nnj.skipconnection import SkipConnection

#########################
### parametric layers ###
from stochman.nnj.linear import Linear #jmp, mjp, jTmjp-to-full non optimal wrt weights
from stochman.nnj.poslinear import PosLinear
from stochman.nnj.conv1d import Conv1d
from stochman.nnj.conv2d import Conv2d
from stochman.nnj.conv3d import Conv3d
from stochman.nnj.convtranspose1d import ConvTranspose1d
from stochman.nnj.convtranspose2d import ConvTranspose2d
from stochman.nnj.convtranspose3d import ConvTranspose3d
from stochman.nnj.batchnorm1d import BatchNorm1d #missing almost everything
from stochman.nnj.batchnorm2d import BatchNorm2d #missing almost everything
from stochman.nnj.batchnorm3d import BatchNorm3d #missing almost everything

#############################
### non-parametric layers ###

# shape preserving, diagonal jacobian
from stochman.nnj.tanh import Tanh
from stochman.nnj.relu import ReLU
from stochman.nnj.sigmoid import Sigmoid
from stochman.nnj.prelu import PReLU
from stochman.nnj.elu import ELU
from stochman.nnj.hardshrink import Hardshrink
from stochman.nnj.hardtanh import Hardtanh
from stochman.nnj.leakyrelu import LeakyReLU
from stochman.nnj.softplus import Softplus
from stochman.nnj.arctanh import ArcTanh
from stochman.nnj.reciprocal import Reciprocal
from stochman.nnj.oneminusx import OneMinusX
from stochman.nnj.sqrt import Sqrt
from stochman.nnj.flatten import Flatten
from stochman.nnj.reshape import Reshape

# shape preserving, non-diagonal jacobian
from stochman.nnj.l2norm import L2Norm
from stochman.nnj.softmax import Softmax #only jacobian and jmp

# non shape preserving
from stochman.nnj.upsample import Upsample
from stochman.nnj.maxpool1d import MaxPool1d #only jmp
from stochman.nnj.maxpool2d import MaxPool2d #missing vjp, jvp
from stochman.nnj.maxpool3d import MaxPool3d #only jmp
from stochman.nnj.rbf import RBF


#old implementation in softmax, batchnorm <-- need to fix the shapes!