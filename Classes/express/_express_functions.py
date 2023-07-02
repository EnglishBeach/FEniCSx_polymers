import dolfinx as _dolfinx
import ufl as _ufl
from classes import Base as _Base


def make_variables(domain: _dolfinx.mesh.Mesh,
    variables: str,
    variable_elements: dict,
    n_variable_versions: int,
):
    """
    Make functions and variable, which can used in study

    Args:
        domain (dolfinx.mesh.Mesh): Domain
        variables (str): Comm! sepatated varables string 'a,b,c,d'
        variable_elements (dict): ufl.Elenent for every variable {'a':ufl.Element}
        n_variable_versions (int): How much steps method use to time discretization
        or how mush similar functions you need (a,a1,a2,a3...)

    Returns (tuple):
        {   'space': fem.Space,
            'subspaces': [list of subspaces],
            'coordinates': [3 spatial coordinates]}
        [list of dicts for every time step in method:
        {   'function': fem.Function,
            'subfunctions': [list of subfunctions],
            'variables': [list of variables connected with subfunctions],
            'version': variable version number,
        }]
    """

    variables = list(variables.strip().split(','))
    assert len(variables) == len(set(variables))

    space = _Base.FunctionSpace(
        mesh=domain,
        element=_ufl.MixedElement(*[variable_elements[var]
                                   for var in variables]),
    )
    subspace = [
        space.sub(variables.index(var)).collapse()[0] for var in variables
    ]
    space_info = {
        'space': space,
        'subspaces': subspace,
        'coordinates': (list(_ufl.SpatialCoordinate(space)) + [None] * 2)[:3],
    }

    function_info = []
    for version in range(n_variable_versions-1, -1, -1):

        step_info = {'version': version}

        func = _Base.Function(space)
        step_info.update({'function': func})

        sun_func = [func.sub(variables.index(var)) for var in variables]
        step_info.update({'subfunctions': sun_func})

        func_splitted = _Base.split(func)
        func_variable = [
            func_splitted[variables.index(var)] for var in variables
        ]
        step_info.update({'variables': func_variable})

        function_info.append(step_info)
    return space_info, function_info

def create_facets(domain):
    _Base.set_connectivity(domain)
    ds = _Base.Measure("ds", domain=domain)
    return ds