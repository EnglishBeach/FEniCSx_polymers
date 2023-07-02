from ufl import FacetNormal,Measure,SpatialCoordinate
from ufl import TrialFunction, TestFunction, TrialFunctions, TestFunctions
from ufl import conditional
from ufl import variable
from ufl import diff as D
from ufl import split, nabla_div, nabla_grad, grad, div
from ufl import as_matrix as matrix
from ufl import exp, sym, tr, sqrt, ln, sin, cos
from matplotlib import pyplot as plt
from ufl import dx
from dolfinx import mesh
from dolfinx.fem import FunctionSpace
from mpi4py import MPI