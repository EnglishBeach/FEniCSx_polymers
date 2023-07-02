import dolfinx as _dolfinx
import ufl as _ufl
import classes.fenics as _fn
import numpy as _np
import matplotlib.pyplot as _plt


def make_variables(
    domain: _dolfinx.mesh.Mesh,
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

    space = _fn.FunctionSpace(
        mesh=domain,
        element=_ufl.MixedElement(
            *[variable_elements[var] for var in variables]),
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
    for version in range(n_variable_versions - 1, -1, -1):

        step_info = {'version': version}

        func = _fn.Function(space)
        step_info.update({'function': func})

        sun_func = [func.sub(variables.index(var)) for var in variables]
        step_info.update({'subfunctions': sun_func})

        func_splitted = _fn.split(func)
        func_variable = [
            func_splitted[variables.index(var)] for var in variables
        ]
        step_info.update({'variables': func_variable})

        function_info.append(step_info)
    return space_info, function_info


def create_facets(domain):
    _fn.set_connectivity(domain)
    ds = _fn.Measure("ds", domain=domain)
    return ds


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