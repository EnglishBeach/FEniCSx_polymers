"""
Standart parametrs for study.
"""

import numpy as _np
from fenics.express import Parametr_container


class solver_confs(Parametr_container):
    """
    Parametrs from Base.Nonlinear_problem
    """

    solve_options = {
        'convergence': 'incremental',
        'tolerance': 1E-6,
    }
    petsc_options = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps'
    }
    form_compiler_options = {}
    jit_options = {}


class mesh_confs(Parametr_container):
    left = 0
    right = 1
    intervals = 100
    degree = 1
    family = 'CG'


class rates(Parametr_container):
    general = 0.01
    P_step = 0.13
    a = 0.1
    b = 1
    e = 1
    gamma = 4


class light_confs(Parametr_container):
    type = 'stepwise'
    left = 0.4
    right = 0.6
    smoothing = 100


class initial(Parametr_container):
    n = 0.2
    p = 0.001


class time(Parametr_container):
    line = _np.linspace(0, 1, 101)
    check = line[::10]


class save_confs(Parametr_container):
    file_name = 'solve'
    dir = '/home/Solves/'

    def __init__(self, name, desc):
        self.name = self._input(name)
        self.desc = self._input(desc)


class Data(Parametr_container):
    solver_confs = solver_confs()
    mesh_confs = mesh_confs()
    bcs = {'type': 'close'}
    time = time()

    rates = rates()
    light_confs = light_confs()
    initial = initial()

    def __init__(self, *args, **kwargs) -> None:
        self.save_confs = save_confs(*args, **kwargs)
