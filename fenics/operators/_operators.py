"""
Base module for simple set up study and solving
"""
import numpy as _np

import ufl as _ufl
from dolfinx import fem as _fem
from dolfinx import nls as _nls

from dolfinx import mesh
from mpi4py import MPI
from ufl import FacetNormal, Measure, SpatialCoordinate
from ufl import TrialFunction, TestFunction, TrialFunctions, TestFunctions
from ufl import conditional
from ufl import variable
from ufl import diff as D
from ufl import split, nabla_div, nabla_grad, grad, div
from ufl import as_matrix as matrix
from ufl import exp, sym, tr, sqrt, ln, sin, cos
from ufl import dx
from dolfinx.fem import FunctionSpace


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
And = _Infix(_ufl.And)


# Methods:
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
        facets = mesh.locate_entities_boundary(
            domain,
            domain.topology.dim - 1,
            condition,
        )
        facet_indices.append(facets)
        facet_markers.append(_np.full_like(facets, marker))
    facet_indices = _np.hstack(facet_indices).astype(_np.int32)
    facet_markers = _np.hstack(facet_markers).astype(_np.int32)
    sorted_facets = _np.argsort(facet_indices)
    facet_tags = mesh.meshtags(
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


def create_facets(domain):
    set_connectivity(domain)
    ds = Measure("ds", domain=domain)
    return ds


# Functions
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
        func = _fem.Function(space)
        if name is not None: func.name = name
        if form is None: return func

        form_type = str(form.__class__)[8:-2]
        cords = _ufl.SpatialCoordinate(space)

        if form_type == ('dolfinx.fem.function.Function'):
            expression = Function.from_fem(form)

        elif form_type == ('dolfinx.fem.function.Constant'):
            expression = Function.from_constant(space, cords, form)

        elif form_type[:3] == 'ufl':
            expression = Function.from_ufl(space, cords, form)

        elif form_type == 'function':
            expression = Function.from_function(form)

        elif form_type == ('dolfinx.fem.function.Expression'):
            expression = Function.from_expression(form)

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
        expression = _fem.Expression(form2,
                                     space.element.interpolation_points())
        return expression

    @staticmethod
    def from_fem(form):
        return form

    @staticmethod
    def from_function(form):
        return form

    @staticmethod
    def from_expression(form):
        return form


class Constant:
    """Constant on space

    Args:
        space (fem.FunctionSpace| domain): Space or domain
        const (auny number): Any number

    Returns:
        fem.function.Constant: Constant on space
    """

    def __new__(
        cls,
        domain_space,
        const,
        const_type=_fem.petsc.PETSc.ScalarType,
    ):

        return _fem.Constant(domain_space, const_type(const))


class DirichletBC:
    """Create Dirichlet condition.

    Args:
    \n space (fem.FunctionSpace): Function space.
    For several spaces: first space is general.
    \n form (any function): Function
    \n combined_marker (Any): One from next:
        \nFunction - boundary marker function find geometrical
        \nAll - all boundary find entities
        \n(mesh.meshtags, marker) -Find entities marker of boundary from mesh tags

    Returns:
        condition (dirichletbc): Dirichlet condition
    """

    def __new__(cls, space, form, combined_marker):
        if isinstance(space, tuple or list): space0 = space[0]
        else: space0 = space
        domain = space0.mesh

        if combined_marker == 'All':
            facets = mesh.exterior_facet_indices(domain.topology)
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

        bc = DirichletBC.set_dirichlet(dofs, form, space0)

        return bc

    @staticmethod
    def set_dirichlet(dofs, form, space):
        if hasattr(form, 'function_space'):
            if form.function_space == space:
                bc = _fem.dirichletbc(dofs=dofs, value=form)
            else:
                bc = _fem.dirichletbc(V=space, dofs=dofs, value=form)
        else:
            bc = _fem.dirichletbc(V=space, dofs=dofs, value=form)
        return bc


# Solvers
from dolfinx.fem.petsc import LinearProblem


class NonlinearProblem:
    """Create nonlinear problem

    Args:
        \n F (ufl.Form): Nonlinear equation form
        \n bcs (Dirichlet): Dirichlet conditious.
        \n u (fem.Function): Function to be solved.
        \n J (ufl.Form): Jacobian matrix. Defaults None.
        \n petsc_options (dict): Options to petsc. Defaults to {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
        }.
        \n solve_options (dict): Options to NEwton solwer.
        Defaults to {'convergence': 'incremental', 'tolerance': 1E-6}.
        \n ghost_opions (dict):  You cant change it
        {'addv': INSERT,'mode': FORWARD}
        \n form_compiler_params (dict): Form compiler options.
        Defaults to {}.
        \n jit_params (dict): JIT parmetrs.
        Defaults to {}.
    """

    # TODO: Make succession
    def __init__(
        self,
        equation: _ufl.Form,
        bcs: list,
        func: _fem.Function,
        Jacobian: _ufl.Form = None,
        solve_options={
            'convergence': 'incremental', 'tolerance': 1E-6
        },
        petsc_options={
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
        },
        form_compiler_options={},
        jit_options={},
    ):
        self._function = func
        self.bcs = bcs

        # Make problem
        problem = _fem.petsc.NonlinearProblem(
            F=equation,
            u=self._function,
            bcs=self.bcs,
            J=Jacobian,
            form_compiler_params=form_compiler_options,
            jit_params=jit_options,
        )
        self._linear_form = problem.a
        self._bilinear_form = problem.L

        # Creating solver
        self._solver = _nls.petsc.NewtonSolver(
            self._function.function_space.mesh.comm,
            problem,
        )
        self.set_options(
            petsc_options=petsc_options,
            solve_options=solve_options,
        )

    def set_options(self, petsc_options, solve_options):
        self._solver.convergence_criterion = solve_options['convergence']
        self._solver.rtol = solve_options['tolerance']

        ksp = self._solver.krylov_solver
        problem_prefix = ksp.getOptionsPrefix()
        options = _fem.petsc.PETSc.Options()
        options.prefixPush(problem_prefix)
        for key, value in petsc_options.items():
            options[key] = value
        ksp.setFromOptions()

    def solve(self):
        """Solve function

        Returns:
            fem.Function: Solved function
        """
        result = self._solver.solve(self._function)
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
    def linear_form(self) -> _fem.FormMetaClass:
        """The compiled linear form"""
        return self._bilinear_form

    @property
    def bilinear_form(self) -> _fem.FormMetaClass:
        """The compiled bilinear form"""
        return self._linear_form


def make_space(
    domain: mesh.Mesh,
    elements: dict,
):
    """
    Make function space and subspaces from dict of element types

    Args:
        domain (dolfinx.mesh.Mesh): Domain
        elements (dict): {'variable' : element type}

    Returns:
        dict: {'space': fem.MixedFunctionSpace, 'subspaces': [fem.Subspace] , 'coordinates' : SpatialCoordinate}
    """
    space = FunctionSpace(
        mesh=domain,
        element=_ufl.MixedElement(*[elements[var] for var in elements]),
    )

    variables = list(elements.keys())
    sub_spaces = [
        space.sub(variables.index(var)).collapse()[0] for var in variables
    ]
    return {
        'space': space,
        'subspaces': sub_spaces,
        'coordinates': (list(_ufl.SpatialCoordinate(space)) + [None] * 2)[:3],
    }


def make_functions(
    space: _fem.FunctionSpace,
    variables: list,
    function_index: int,
):
    """
    Make functions and their components on function space

    Args:
        space (fem.FunctionSpace): Sunction space
        variables (list): List of variables
        function_index (int): Function index

    Returns:
        dict: {'function': fem.Function, 'sub_functions': [fem.Subfunction],'variables': [subfunction indexes]}
    """
    func = Function(space)
    func.name = 'Function_' + str(function_index)

    sub_funcs = [func.sub(variables.index(var)) for var in variables]
    for sub_func, var in zip(sub_funcs, variables):
        sub_func.name = var + str(function_index)

    func_splitted = split(func)
    func_variables = [func_splitted[variables.index(var)] for var in variables]
    return {
        'function': func,
        'subfunctions': sub_funcs,
        'variables': func_variables,
    }