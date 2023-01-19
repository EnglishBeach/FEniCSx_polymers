# Base module
import typing as _typing
import shutil as _shutil
import re as _re
import numpy as _np

import ufl as _ufl
import dolfinx as _dolfinx
from dolfinx import mesh as _mesh
from dolfinx import fem as _fem
from dolfinx import nls as _nls
from ufl import FacetNormal, SpatialCoordinate, Measure
from ufl import TrialFunction, TestFunction, TrialFunctions, TestFunctions, dx
from ufl import conditional
from ufl import variable
from ufl import diff as D
from ufl import nabla_div, nabla_grad, grad, div
from ufl import as_matrix as matrix
from ufl import lhs, rhs, split
from ufl import exp, sym, tr, sqrt, ln, sin, cos
from dolfinx.fem import FunctionSpace
from matplotlib import pyplot as _plt


# Operators
class _Infix:
    """Create infix function from default"""

    def __init__(self, function):
        self.function = function

    def __ror__(self, other):
        return _Infix(lambda x, self=self, other=other: self.function(other, x))

    def __or__(self, other):
        return self.function(other)

    def __call__(self, value1, value2):
        return self.function(value1, value2)


dot = _Infix(_ufl.dot)
inner = _Infix(_ufl.inner)

npor = _Infix(_np.logical_or)
npand = _Infix(_np.logical_and)


def vector(*args):
    return _ufl.as_vector(tuple(args))


def I(func_like):
    """Create matrix Identity dimension of func_like

    Args:
        func_like (Function): Give geometric dimension

    Returns:
        Tensor: Identity
    """
    return _ufl.Identity(func_like.geometric_dimension())


# Functions:
class ArrayFunc:

    def __init__(self, func: _fem.Function, name=None):
        self.cord = _np.array([
            func.function_space.tabulate_dof_coordinates()[:, 0],
            func.x.array,
        ])
        self._sort()
        self.cord = _np.array([[a, b]for a, b in enumerate(self.cord[1])]).transpose() #yapf: disable
        self.len = len(self.cord[0])
        self.name = name

    def _sort(self):
        self.cord = self.cord[:, _np.argsort(self.cord[0])]

    def translate(self, point_0):
        x_new = (self.cord[0] - point_0)
        self.cord[0] = x_new - self.len * (x_new // (self.len))
        self._sort()

    def mirror(self, point_0):
        self.translate(point_0)
        self.cord[0] = -self.cord[0]
        self.cord[0] += self.len
        self._sort()


def get_space_dim(space):
    """Get dimensions of X on space

    Args:
        space (fem.FunctionSpace): Space

    Returns:
        List: space dim, len
    """
    return (space.mesh.geometry.dim, len(space.dofmap.list.array))


def create_FacetTags_boundary(domain, bound_markers):
    """Mark boundary facets under conditious

    Args:
        domain (Domain): Domain
        bound_markers (mark,python_function): List of mark and function

    Return:
        tags(mesh.meshtags): Marked facets
    """
    facet_indices, facet_markers = [], []
    for (marker, condition) in bound_markers:
        facets = _mesh.locate_entities_boundary(
            domain,
            domain.topology.dim - 1,
            condition,
        )
        facet_indices.append(facets)
        facet_markers.append(_np.full_like(facets, marker))
    facet_indices = _np.hstack(facet_indices).astype(_np.int32)
    facet_markers = _np.hstack(facet_markers).astype(_np.int32)
    sorted_facets = _np.argsort(facet_indices)
    facet_tags = _mesh.meshtags(
        domain,
        domain.topology.dim - 1,
        facet_indices[sorted_facets],
        facet_markers[sorted_facets],
    )

    return facet_tags


def set_connectivity(domain):
    """Need to compute facets to Boundary value

    Args:
        domain (Mesh): Domain
    """
    domain.topology.create_connectivity(
        domain.topology.dim - 1,
        domain.topology.dim,
    )


# Classes
class DirichletBC:
    """
    Create Dirichlet condition.

    Args:
        space (fem.FunctionSpace): Function space.
        For several spaces:: first space is general.
        form (any function): Function
        combined_marker (Any): One from next::
        \nFunction - boundary marker function find geometrical
        \nAll - all boundary find entities
        \n(mesh.meshtags, marker) -Find entities marker of boundary from mesh tags

    Returns:
        condition (dirichletbc): Dirichlet condition
    """

    def __new__(cls, space, form, combined_marker):

        def set_dirichlet(dofs, form, space):
            if hasattr(form, 'function_space'):
                if form.function_space == space:
                    bc = _fem.dirichletbc(dofs=dofs, value=form)
                else:
                    bc = _fem.dirichletbc(V=space, dofs=dofs, value=form)
            else:
                bc = _fem.dirichletbc(V=space, dofs=dofs, value=form)
            return bc

        # FIXME: Maybe listable?
        if isinstance(space, tuple or list): space0 = space[0]
        else: space0 = space
        domain = space0.mesh

        if combined_marker == 'All':
            facets = _mesh.exterior_facet_indices(domain.topology)
            dofs = _fem.locate_dofs_topological(
                space,
                domain.topology.dim - 1,
                facets,
            )

        elif isinstance(combined_marker, tuple or list):
            marked_facets, marker = combined_marker
            facets = marked_facets.find(marker)
            dofs = _fem.locate_dofs_topological(
                space,
                domain.topology.dim - 1,
                facets,
            )

        else:
            dofs = _fem.locate_dofs_geometrical(space, combined_marker)

        bc = set_dirichlet(dofs, form, space0)

        return bc


class Function:
    """Function on new space. Default = 0

    Args:
        space (FunctionSpace): Function space
        form (): Any form: fem.Function,fem.Constant,fem.Expression,
        ufl_function, callable function, number

    Returns:
        fem.Function: Function
    """

    def __new__(cls, space, form=None, name=None):

        # FIXME: x-x not beauty

        func = _fem.Function(space)
        if name is not None: func.name = name
        if form is None: return func

        form_type = str(form.__class__)[8:-2]
        cords = _ufl.SpatialCoordinate(space)
        
        if form_type == ('dolfinx.fem.function.Function'):
            expression = Function.from_fem(space, cords, form)

        elif form_type == ('dolfinx.fem.function.Constant'):
            expression = Function.from_constant(space, cords, form)

        elif form_type[:3] == 'ufl':
            expression = Function.from_ufl(space, cords, form)

        elif form_type == 'function':
            expression = Function.from_function(space, cords, form)

        elif form_type == ('dolfinx.fem.function.Expression'):
            expression = Function.from_expression(space, cords, form)

        elif not callable(form):
            expression = Function.from_number(space, cords, form)

        else:
            raise ValueError(f'Uncorrect form:{form_type}')

        func.interpolate(expression)
        return func

    @staticmethod
    def from_constant(space, cords, form):
        if len(form.ufl_shape) == 0:
            form2 = form.value + (cords[0] - cords[0])
        else:
            form2 = vector(*form.value)
            form2 += vector(*map(lambda x, y: x - y, cords, cords))
        expression = _fem.Expression(
            form2,
            space.element.interpolation_points(),
        )
        return expression

    @staticmethod
    def from_ufl(space, cords, form):
        if len(form.ufl_shape) == 0:
            form2 = form + (cords[0] - cords[0])
        else:
            form2 = form + vector(*map(lambda x, y: x - y, cords, cords))
        expression = _fem.Expression(
            form2,
            space.element.interpolation_points(),
        )
        return expression

    @staticmethod
    def from_number(space, cords, form):
        if not hasattr(form, '__getitem__'):
            form2 = form + (cords[0] - cords[0])
        else:
            form2 = vector(*form)
            form2 += vector(*map(lambda x, y: x - y, cords, cords))
        expression = _fem.Expression(
            form2, space.element.interpolation_points()
        )
        return expression

    @staticmethod
    def from_fem(space, cords, form):
        return form

    @staticmethod
    def from_function(space, cords, form):
        return form

    @staticmethod
    def from_expression(space, cords, form):
        return form


class Constant:

    def __new__(cls, domain_space, const):
        """Constant on space

        Args:
            space (fem.FunctionSpace| domain): Space or domain
            const (auny number): Any number

        Returns:
            fem.function.Constant: Constant on space
        """
        return _fem.Constant(domain_space, _fem.petsc.PETSc.ScalarType(const))


# Solvers
from dolfinx.fem.petsc import LinearProblem


class NonlinearProblem:
    """Create nonlinear problem

        Args:
            F (ufl.Form): Nonlinear equation form
            bcs (Dirichlet): Dirichlet conditious.
            u (fem.Function): Function to be solved.
            \nJ (ufl.Form): Jacobian matrix. Defaults None.
            \npetsc_options (dict): Options to petsc. Defaults to {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
            }.
            \nsolve_options (dict): Options to NEwton solwer.
            Defaults to {'convergence': 'incremental', 'tolerance': 1E-6}.
            \nghost_opions (dict):  You cant change it
            {'addv': INSERT,'mode': FORWARD}
            \nform_compiler_params (dict): Form compiler options.
            Defaults to {}.
            \njit_params (dict): JIT parmetrs.
            Defaults to {}.
        """

    # TODO: Make succession
    def __init__(
        self,
        F: _ufl.Form,
        bcs: list,
        u: _fem.Function,
        J: _ufl.Form = None,
        solve_options={
            'convergence': 'incremental', 'tolerance': 1E-6
        },
        petsc_options={
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
        },
        form_compiler_params={},
        jit_params={},
    ):

        def set_options(self, petsc_options, solve_options):
            self._solver.convergence_criterion = solve_options['convergence']
            self._solver.rtol = solve_options['tolerance']

            ksp = self._solver.krylov_solver
            problem_prefix = ksp.getOptionsPrefix()
            opts = _fem.petsc.PETSc.Options()
            opts.prefixPush(problem_prefix)
            for k, v in petsc_options.items():
                opts[k] = v
            ksp.setFromOptions()

        self._u = u
        self.bcs = bcs

        pr = _fem.petsc.NonlinearProblem(
            F=F,
            u=self._u,
            bcs=self.bcs,
            J=J,
            form_compiler_params=form_compiler_params,
            jit_params=jit_params,
        )
        self._a = pr.a
        self._L = pr.L

        # Creating solver
        self._solver = _nls.petsc.NewtonSolver(
            self._u.function_space.mesh.comm,
            pr,
        )
        set_options(
            self, petsc_options=petsc_options, solve_options=solve_options
        )

    def solve(self):
        """Solve function

        Returns:
            fem.Function: Solved function
        """
        result = self._solver.solve(self._u)
        return result

    @staticmethod
    def KSP_types():
        """Get KSP types"""
        return _fem.petsc.PETSc.KSP.Type

    @staticmethod
    def PC_types():
        """Get PC types"""
        return _fem.petsc.PETSc.PC.Type

    @property
    def solver(self) -> _fem.petsc.PETSc.KSP:
        """Linear solver object"""
        return self._solver

    @property
    def L(self) -> _fem.FormMetaClass:
        """The compiled linear form"""
        return self._L

    @property
    def a(self) -> _fem.FormMetaClass:
        """The compiled bilinear form"""
        return self._a


# Postprocess
def func_plot1D(
    funcs: list,
    fig=None,
    ax=None,
    show_points=False,
):
    """Create plot from fem.Function
    Args:
        funcs (fem.Function, str): list of functions
        show_points (bool): To show points on plot
        fig (plt.Figure): Figure to go back it
        ax (plt.axes): Axes to go back it
    """
    if (fig or ax) is None:
        fig, ax = _plt.subplots(facecolor='White')
        fig.set_size_inches(16, 8)
    for func in funcs:
        x = func.function_space.tabulate_dof_coordinates()[:, 0]
        y = func.x.array
        cord = _np.array([x, y])
        cord = cord[:, _np.argsort(cord[0])]
        ax.plot(cord[0], cord[1], label=func.name, linewidth=1)
        if show_points: ax.scatter(cord[0], cord[1], s=0.5)
    ax.legend(
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0,
        loc='center left',
    )
    return ax


def func_plot2D(
    func,
    show_points=False,
    smooth_show=True,
    fig=None,
    ax=None,
):
    """Create plot from fem.Function
    Args:
        func (fem.Function, str): Function
        show_points (bool): To show real function mesh
        smooth_show (bool): To show smoth plot instead real plot
        fig (plt.Figure): Figure to go back it
        ax (plt.axes): Axes to go back it
    """
    if (fig or ax) is None:
        fig, ax = _plt.subplots(facecolor='White')
        fig.set_size_inches(10, 10)

    data = _np.column_stack((
        func.function_space.tabulate_dof_coordinates()[:, 0:2],
        func.x.array,
    ))
    data = data.transpose()

    y = func.x.array
    if smooth_show:
        plot = ax.tripcolor(*data)
    else:
        levels = _np.linspace(func.x.array.min(), func.x.array.max(), 10)
        plot = ax.tricontourf(*data, levels=levels)
    if show_points: ax.plot(data[0], data[1], 'o', markersize=2, color='grey')
    fig.colorbar(plot, ax=ax)
    return ax

    # FIXME


# class LinearProblem:
#     """Create linear  problem

#         Args:
#             a (ufl.Form): bilinear form
#             L (ufl.Form): linear form
#             bcs (Dirichlet): Dirichlet conditious.
#             u (fem.Function): Function to be solved.
#             \npetsc_options (dict): Options to petsc.
#             Defaults to { 'ksp_type': 'preonly', 'pc_type': 'lu' }.
#             \nassemble_options (dict): Options to assemble bilinear and linear forms.
#             Defaults to {'assebmle_A': True, 'assemble_B': True}.
#             \nghost_opions (dict): GhostUpdate potions.
#             Defaults to  {'addv': ADD,'mode': REVERSE}.
#             \nform_compiler_params (dict): Form compiler options.
#             Defaults to {}.
#             \njit_params (dict): JIT parmetrs.
#             Defaults to {}.
#         """

#     def __init__(
#         self,
#         a: _ufl.Form,
#         L: _ufl.Form,
#         bcs: list,
#         u: _fem.Function,
#         petsc_options={
#             'ksp_type': 'preonly', 'pc_type': 'lu'
#         },
#         assemble_options={
#             'assemble_A': True, 'assemble_b': True
#         },
#         ghost_opions={},
#         form_compiler_params={},
#         jit_params={},
#     ):
#         # FIXME: Maybe need setiings options to forms or not?
#         def set_options(self, petsc_options):
#             ksp = self._solver
#             problem_prefix = f'dolfinx_solve_{id(self)}'
#             ksp.setOptionsPrefix(problem_prefix)
#             opts = _fem.petsc.PETSc.Options()
#             opts.prefixPush(problem_prefix)
#             for k, v in petsc_options.items():
#                 opts[k] = v
#             opts.prefixPop()
#             ksp.setFromOptions()
#             # self._A.setOptionsPrefix(problem_prefix)
#             # self._A.setFromOptions()
#             # self._b.setOptionsPrefix(problem_prefix)
#             # self._b.setFromOptions()

#         self._u = u
#         self.bcs = bcs

#         # A form
#         self._a = _fem.form(
#             a,
#             form_compiler_params=form_compiler_params,
#             jit_params=jit_params,
#         )
#         self._A = _fem.petsc.create_matrix(self._a)

#         # b form
#         self._L = _fem.form(
#             L,
#             form_compiler_params=form_compiler_params,
#             jit_params=jit_params,
#         )
#         self._b = _fem.petsc.create_vector(self._L)

#         # Creating solver
#         self._solver = _fem.petsc.PETSc.KSP().create(
#             self._u.function_space.mesh.comm
#         )
#         self._solver.setOperators(self._A)
#         set_options(self, petsc_options)

#         # Another options
#         self._ghost_opions = {
#             'addv': _fem.petsc.PETSc.InsertMode.ADD,
#             'mode': _fem.petsc.PETSc.ScatterMode.REVERSE,
#         }
#         self._ghost_opions.update(ghost_opions)

#         # Assembling
#         self.assemble_options = assemble_options
#         if self.assemble_options['assemble_A']: self._assemble_A()
#         if self.assemble_options['assemble_b']: self._assemble_b()

#     def _assemble_A(self):
#         """Assemle bilinear form"""
#         self._A.zeroEntries()
#         _fem.petsc._assemble_matrix_mat(self._A, self._a, bcs=self.bcs)
#         self._A.assemble()

#     def _assemble_b(self):
#         """Assemble linear form"""
#         with self._b.localForm() as b_loc:
#             b_loc.set(0)
#         _fem.petsc.assemble_vector(self._b, self._L)
#         _fem.petsc.apply_lifting(self._b, [self._a], bcs=[self.bcs])
#         self._b.ghostUpdate(
#             addv=self._ghost_opions['addv'],
#             mode=self._ghost_opions['mode'],
#         )
#         _fem.petsc.set_bc(self._b, self.bcs)

#     def solve(self):
#         """Solve function

#         Returns:
#             fem.Function: Solved function
#         """
#         if not self.assemble_options['assemble_A']: self._assemble_A()
#         if not self.assemble_options['assemble_b']: self._assemble_b()

#         result = self._solver.solve(self._b, self._u.vector)
#         self._u.x.scatter_forward()
#         return result

#     @staticmethod
#     def KSP_types():
#         """Get KSP types"""
#         return _fem.petsc.PETSc.KSP.Type

#     @staticmethod
#     def PC_types():
#         """Get PC types"""
#         return _fem.petsc.PETSc.PC.Type

#     @staticmethod
#     def ghost_updates():
#         """Get ghost_update types"""
#         return (_fem.petsc.PETSc.InsertMode, _fem.petsc.PETSc.ScatterMode)

#     @property
#     def L(self) -> _fem.FormMetaClass:
#         """The compiled linear form"""
#         return self._L

#     @property
#     def a(self) -> _fem.FormMetaClass:
#         """The compiled bilinear form"""
#         return self._a

#     @property
#     def A(self) -> _fem.petsc.PETSc.Mat:
#         """Matrix operator"""
#         return self._A

#     @property
#     def b(self) -> _fem.petsc.PETSc.Vec:
#         """Right-hand side vector"""
#         return self._b

#     @property
#     def solver(self) -> _fem.petsc.PETSc.KSP:
#         """Linear solver object"""
#         return self._solver