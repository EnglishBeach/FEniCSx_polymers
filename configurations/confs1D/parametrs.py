"""
Standart parametrs for study.
"""

import numpy as _np
import jsonpickle as _jp


class Parametr_container:
    def __repr__(self):
        key_list = [
            f'{compound_key}: {value}' for compound_key,
            value in self.Options.items()
        ]
        return str('\n'.join(key_list))

    @property
    def Options(self):
        return {
            compound_key.strip(): value
            for value,
            compound_key in self._recursion_view()
        }

    def _all_options(self):
        options = list(
            filter(lambda x: (x[0] != '_') and x[0].islower(), self.__dir__()))
        return {option: self.__getattribute__(option) for option in options}

    def _recursion_view(self, composite_key=''):
        if isinstance(self, Parametr_container):
            for key in self._all_options():
                for inner_key in Parametr_container._recursion_view(
                        self.__getattribute__(key),
                        str(composite_key) + '  ' + str(key),
                ):
                    yield inner_key
        elif isinstance(self, dict):
            for key in self:
                for inner_key in Parametr_container._recursion_view(
                        self[key],
                        str(composite_key) + '  ' + str(key),
                ):
                    yield inner_key
        elif isinstance(self, _np.ndarray|range):
            yield (
                f'[{self[0]}, {self[1]} .. {self[-2]}, {self[-1]}]; len = {len(self)}',
                composite_key)
        else:
            yield (self, composite_key)

    @staticmethod
    def _input(parametr):
        if parametr is None:
            while parametr != 'q':
                parametr = input(f'Set {parametr=}, to quit - q:')
        return parametr


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
