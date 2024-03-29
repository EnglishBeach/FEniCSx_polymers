import re as _re
import numpy as _np

from matplotlib.ticker import FormatStrFormatter as _FormatStrFormatter
import matplotlib.pyplot as _plt
import ufl as _ufl
import dolfinx as _dolfinx
from dolfinx import fem as _fem

from .. import operators as _fn


def get_view(func, variables):
    variables = list(variables.strip().split(','))
    string = str(func)
    replacement = {
        r'\({ A \| A_{.*?} = (.+?)}\)': r'\n\1',
        r'(d[sx])\(.*?\)(?:\\n)?': r'\1',
        r'Function_([\d+])\[(\d+)\]': r'var{\2_\1}',
        r'\+ ?-(\d+)': r'-\1',
        r'\* ?1 |(?<=[\D])1 ?\* ?': '',
        r'\( ?grad ?\((.*?)\) ?\)': r'grad(\1)',
        r'v_\d+\[(\d+)\]': r'testvar{\1}',
        r'\[i_.*?\]': '',
        r' +': ' ',
    }
    for key in replacement:
        string = _re.sub(
            string=string,
            pattern=key,
            repl=replacement[key],
        )

    vars_indexes = range(len(variables))

    for i in vars_indexes:

        string = _re.sub(
            string=string,
            pattern=r'var{' + str(i) + r'_(\d+)}',
            repl=str(variables[i]) + r'\1',
        )
        string = _re.sub(
            string=string,
            pattern=r'testvar{{' + str(i) + r'}',
            repl='test_' + variables[i],
        )
    return string


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


class ArrayFunc:

    def __init__(self, func, name=None):

        if isinstance(func, _fem.Function):
            x_line = func.function_space.tabulate_dof_coordinates()[:, 0]
            y_line = func.x.array
            self.name = func.name
        else:
            x_line = func[:, 0]
            y_line = func[:, 1]

        self.name = name

        self._data = _np.array([x_line, x_line, y_line]).T
        self.cord, self.x, self.y = self[:, 0], self[:, 1], self[:, 2]

        self.sort()
        self.cord[:] = range(len(x_line))

    def __len__(self):
        return len(self.x)

    def __repr__(self):
        string = f"""ArrayFunc len = {len(self)} :\nNum     X             Y\n"""
        point_str = '\n'.join([
            f'{int(point[0]):<5}  {point[1]:>5e}  {point[2]:<5e}'
            for point in self._data
        ])

        return string + point_str

    def __getitem__(self, position):
        return self._data[position]

    def __setitem__(self, i, value):
        self._data[i] = value

    def copy(self):
        coped = ArrayFunc(_np.array([self.cord, self.cord]).T)
        coped[:] = self[:]
        coped.name = self.name
        return coped

    def sort(self):
        self._data[:] = _np.array(sorted(self._data, key=lambda x: x[0]))

    def translate(self, point_0):
        new_func = self.copy()

        cord_new = (new_func.cord - point_0)
        new_func.cord[:] = cord_new - len(new_func) * (cord_new //
                                                       (len(new_func)))
        new_func.sort()
        return new_func

    def reverse(self):
        new_func = self.copy()

        new_func.cord[:] = -new_func.cord
        new_func.cord[:] += len(new_func)
        new_func.sort()
        return new_func

    def mirror(self, point_0):
        new_func = self.translate(point_0)
        new_func = new_func.reverse()
        # new_func = new_func.translate(point_0)
        new_func.sort()
        return new_func

    def _find_middle_cord(self, x_middle, add_point):
        middle_cord = int(
            (min(self.x) + x_middle) / (max(self.x) - min(self.x)) * len(self))
        middle_cord += add_point
        return middle_cord

    def _array_plots(self, cord_middle):
        fig, ax = _plt.subplots(facecolor='White')
        fig.set_size_inches(20, 10)
        ax.grid(True, which='Both')
        ax.set_xlim((min(self.cord), max(self.cord)))
        ax.set_ylim((min(self.y), max(self.y)))
        ax.yaxis.set_major_formatter(_FormatStrFormatter(('%.3f')))
        _plt.plot(
            [cord_middle, cord_middle],
            (min(self.y), max(self.y)),
            c='red',
        )

        ax.plot(self.cord, self.y, label=self.name, color='black')
        ax.fill_between(
            self.cord,
            self.y * 0,
            self.y,
            where=self.y > self.y * 0,
            alpha=0.5,
            facecolor='green',
            label=self.name,
        )
        ax.fill_between(
            self.cord,
            self.y * 0,
            self.y,
            where=self.y < self.y * 0,
            alpha=0.5,
            facecolor='red',
            label=self.name,
        )

    def check_symmetry(self, x_middle, point_add=0):

        dif = self.copy()
        cord_middle = self._find_middle_cord(x_middle, point_add)

        dif = dif.mirror(cord_middle)
        dif = dif.translate(-cord_middle)

        aver_y = max(self.y) - min(self.y)
        dif.y[:] = (dif.y - self.y) / aver_y * 100
        dif.y[dif.cord >= cord_middle] = 0

        print(f'Rule: Right - left at x={x_middle}, point = {cord_middle}')
        dif._array_plots(cord_middle)
        self._array_plots(cord_middle)


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


class ParameterClass:

    def __repr__(self):
        key_list = []
        for composite_key, value in self.Options.items():
            add_str = f'{composite_key}: '

            if not isinstance(value, _np.ndarray|range):
                value_str = f'{value}'
            else:
                value_str = f'[{value[0]}, {value[1]} .. {value[-2]}, {value[-1]}]; len = {len(value)}'

            key_list.append(add_str + value_str)
        return str('\n'.join(key_list))

    @property
    def Options(self):
        return {
            compound_key.strip(): value
            for value,
            compound_key in self._recursion_view()
        }

    def __call__(self, *args, **kwds):
        self._add_option(*args, **kwds)

    def _add_option(self, add_options: dict):
        for key, value in add_options.items():
            self.__setattr__(key, value)

    def _all_options(self):
        options = list(
            filter(lambda x: (x[0] != '_') and x[0].islower(), self.__dir__()))
        return {option: self.__getattribute__(option) for option in options}

    def _recursion_view(self, composite_key=''):
        if isinstance(self, ParameterClass):
            for key in self._all_options():
                for inner_key in ParameterClass._recursion_view(
                        self.__getattribute__(key),
                        str(composite_key) + '  ' + str(key),
                ):
                    yield inner_key
        # Dicts store as dicts and include in composite key
        # elif isinstance(self, dict):
        #     for key in self:
        #         for inner_key in Parametr_container._recursion_view(
        #                 self[key],
        #                 str(composite_key) + '  ' + str(key),
        #         ):
        #             yield inner_key
        else:
            yield (self, composite_key)

    @staticmethod
    def _input(parametr):
        if parametr is None:
            while parametr != 'q':
                parametr = input(f'Set {parametr=}, to quit - q:')
        return parametr
