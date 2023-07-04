"""
Standart parametrs for study.
"""

import numpy as _np
import jsonpickle as _jp


class confs:

    def _get_attributes__(self):
        names = list(filter(lambda x: ('__' not in x) and x[0].islower(), self.__dir__()))
        return {name: self.__getattribute__(name) for name in names}

    @property
    def Info(self):
        return {
            compound_key.strip(): value for value,
            compound_key in self._recursion_view__()
        }

    def Get_confs(self):
        key_list = [
            f'{compound_key}: {value}' for
            compound_key,value in self.Info.items()
        ]
        return '\n'.join(key_list)

    def _recursion_view__(self, keys=''):
        if isinstance(self, confs):
            for key in self._get_attributes__():
                for inner_key in confs._recursion_view__(
                        self.__getattribute__(key),
                        str(keys) + '  ' + str(key),
                ):
                    yield inner_key
        elif isinstance(self, _np.ndarray|range):
            yield (
                f'[{self[0]}, {self[1]} .. {self[-2]}, {self[-1]}]; len = {len(self)}',
                keys)
        else:
            yield (self, keys)


class solver_confs(confs):
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


class mesh_confs(confs):
    left = 0
    right = 1
    intervals = 100
    degree = 1
    family = 'CG'


class rates(confs):
    general = 0.01
    P_step = 0.13
    a = 0.1
    b = 1
    e = 1
    gamma = 4


class light_confs(confs):
    type = 'stepwise'
    left = 0.4
    right = 0.6
    smoothing = 100


class initial(confs):
    n = 0.2
    p = 0.001


class time(confs):
    line = _np.linspace(0, 1, 101)
    check = line[::10]


class save_confs(confs):
    file_name = 'solve'
    dir = '/home/Solves/'

    def __init__(self, name=None, desc=None):
        if name is not None:
            self.solution_name = name
            self.description = desc
        else:
            while True:
                self.solution_name = input('Set name')
                if self.solution_name != '': break

            while True:
                self.description = input('Set description')
                if self.description != '': break


class Data(confs):
    solver_confs = solver_confs()
    mesh_confs = mesh_confs()
    bcs = {'type': 'close'}
    time = time()

    rates = rates()
    light_confs = light_confs()
    initial = initial()

    def __init__(self, *args, **kwargs) -> None:
        self.save_confs = save_confs(*args, **kwargs)

    # def dump(self,save=False):

    #     consts = CONST.copy()
    #     DATA.dump.consts = {key: repr_str(value) for key, value in consts.items()}
    #     consts.update({'LIGHT': LIHGT})
    #     DATA.dump.equations = {
    #         key: repr_str(value, consts)
    #         for key, value in EQUATION.items()
    #     }

    #         if not save:
    #             print(DATA.dump.EQUATION_N)
    #             print('*' * 80)
    #             print(DATA.dump.EQUATION_P)
    #         else:
    #             with open(
    #                 self.save_confs.dir_save + self.save_confs.save_name + self.save_confs.file_name +
    #                 '_anotaton.txt',
    #                 'w',
    #             ) as annotation:
    #                 annotation.write(jp.encode(self, numeric_keys=True, indent=4))
    #         pass
