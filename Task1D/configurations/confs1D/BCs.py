import inspect as _inspect
import typing as _typing
from dolfinx import fem as _fem
from fenics import operators as _fn


class BCs1D:

    @classmethod
    def create(cls, style, **parametrs):
        assert style in cls.styles.keys(), 'Not implemented method'
        return getattr(cls, '_' + style)(**parametrs)

    @staticmethod
    def _style(func=None, get=False, l={}):
        if not get:
            args = set(_inspect.getfullargspec(func).args)
            l.update({func.__name__[1:]: ', '.join(args)})
            return func
        else:
            return l

    @classmethod
    @property
    def styles(cls):
        kinds = cls._style(get=True)
        return kinds

    @staticmethod
    @_style
    def _close():
        return 0

    @staticmethod
    @_style
    def _robin(f: _fem.Function, const, ext):
        return -const * (ext-f)

    @staticmethod
    @_style
    def _fixed_flux(value):
        return -value

    @staticmethod
    @_style
    def _like_inside(
        f,
        variables: _typing.List[_fem.Function],
        mesh_interval,
        n_ext,
        p_ext,
        a_nm,
        b_pm,
        e_np,
    ):
        n, p = variables['n'], variables['p']
        if f == n:
            flux_N = 0
            flux_N += -a_nm * (n_ext-n)
            flux_N += +a_nm * p * (n_ext-n)
            flux_N += -e_np * p * (n_ext-n)
            flux_N += -a_nm * n * (p_ext-p)
            flux_N += +e_np * n * (p_ext-p)
            return flux_N / mesh_interval
        elif f == p:
            flux_P = 0
            flux_P += -b_pm * (p_ext-p)
            flux_P += +b_pm * n * (p_ext-p)
            flux_P += -e_np * n * (p_ext-p)
            flux_P += -b_pm * p * (n_ext-n)
            flux_P += +e_np * p * (n_ext-n)
            return flux_P / mesh_interval