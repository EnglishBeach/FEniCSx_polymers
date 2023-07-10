"""
Standart parametrs for study.
"""

import numpy as _np
from fenics.express import ParameterClass


class SolverConfs(ParameterClass):
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


class MeshConfs(ParameterClass):
    left = 0
    right = 1
    intervals = 100
    degree = 1
    family = 'CG'


class Rates(ParameterClass):
    general = 0.01
    P_step = 0.13
    a = 0.1
    b = 1
    e = 1
    gamma = 4


class LightConfs(ParameterClass):
    type = 'stepwise'
    left = 0.4
    right = 0.6
    smoothing = 100


class Initial(ParameterClass):
    n = 0.2
    p = 0.001


class Time(ParameterClass):
    line = _np.linspace(0, 1, 101)
    check = line[::10]


class SaveConfs(ParameterClass):
    file_name = 'solve'
    dir = '/home/Solves/'

    def __init__(self, name, desc):
        self.name = self._input(name)
        self.desc = self._input(desc)


class Data(ParameterClass):
    solver_confs = SolverConfs()
    mesh_confs = MeshConfs()
    bcs = {'type': 'close'}
    time = Time()

    rates = Rates()
    light_confs = LightConfs()
    initial = Initial()

    def __init__(self, *args, **kwargs) -> None:
        self.save_confs = SaveConfs(*args, **kwargs)
