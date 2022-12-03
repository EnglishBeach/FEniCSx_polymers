from FEnicS_base import *
import Parametrs as _pars
from dolfinx import fem as _fem
import ufl as _ufl
import inspect as _inspect
# Task builders
class Comp_INTERFACE:

    def __init__(self, *args, **kwargs):
        self.N = None
        self.P = None
        self.comps = ['N', 'P']
        self._iterate_action(*args, **kwargs)

    def get_num(self, comp):
        return self.comps.index(comp)

    def get_value(self, comp):
        return getattr(self, comp)

    def action(self, comp, *args, **kwargs):
        raise NotImplementedError('No action assigned')

    def _iterate_action(self, *args, **kwargs):
        for comp in self.comps:
            setattr(self, comp, self.action(comp, *args, **kwargs))


class Comp_Element(Comp_INTERFACE):

    def action(self, comp, data: _pars.Param_mesh, domain):
        return _ufl.FiniteElement(
            family=data.family,
            cell=domain.ufl_cell(),
            degree=data.degree,
        )

    def mix(self):
        return _ufl.MixedElement([self.get_value(comp) for comp in self.comps])


class Comp_Sub_spaces(Comp_INTERFACE):

    def action(self, comp, space: _fem.FunctionSpace):
        return space.sub(self.get_num(comp)).collapse()[0]


class Comp_Sub_functions(Comp_INTERFACE):

    def action(self, comp, func: _fem.Function):
        return func.sub(self.get_num(comp))


class Comp_Indicators(Comp_INTERFACE):

    def action(self, comp, func: _fem.Function):
        return split(func)[self.get_num(comp)]


class Task_Consts:

    def __init__(self, indic: Comp_Indicators, data: _pars.Param_const):
        p = indic.P
        n = indic.N

        gen_rate = data.gen_rate
        self.step_P = data.P_step
        self.A_NM = gen_rate * data.a_rate
        self.B_PM = gen_rate * data.b_rate * exp(-p / self.step_P)
        self.E_NP = gen_rate * data.e_rate * exp(-p / self.step_P)

        gamma = data.gamma
        m = 1 - p - n
        under_ln = m / (1-n)
        power = (gamma-1) / gamma
        self.REACTION = gamma * m * (-ln(under_ln))**power


class Light_collection:

    def __init__(self, x: SpatialCoordinate, x0, slope):
        self.x = x
        self.x0 = x0
        self.slope = slope

    def create(self, kind: str):
        assert kind in Light_collection.get_kinds(False), 'Not implemented method'
        return getattr(self, kind)()

    @staticmethod
    def _kind_ready(func=None, get=False, l=set()):
        if not get:
            l.add(func.__name__)
            return func
        else:
            return l

    @classmethod
    def get_kinds(cls, view=True):
        kinds = cls._kind_ready(get=True)
        if not view:
            return kinds
        else:
            print('\n'.join(kinds))

    def _singP(self):
        return (1 + _ufl.sign(self.slope)) / 2

    def _singM(self):
        return (1 - _ufl.sign(self.slope)) / 2

    @_kind_ready
    def step(self):
        return conditional(self.x0 <= self.x, self._singP(), self._singM())

    @_kind_ready
    def sigmoid(self):
        a = self.slope * 5
        return 1 / (1 + exp(-a * (self.x - self.x0)))

    @_kind_ready
    def trapsharp(self):
        a = self.slope
        res = conditional(
            _ufl.And(
                self.x0 - 1 / (abs(2 * a)) <= self.x,
                self.x < self.x0 + 1 / (abs(2 * a))
            ),
            a * (self.x - self.x0) + 0.5,
            0,
        )
        res += conditional(
            self.x0 + 1 / (2*a) <= self.x,
            self._singP(),
            self._singM(),
        )
        return res

    @_kind_ready
    def parab(self):
        a = self.slope * 5
        res = conditional(
            _ufl.And(
                self.x0 - 1 / sqrt(abs(2 * a)) <= self.x,
                self.x < self.x0,
            ),
            a * (self.x - self.x0 + 1 / sqrt(abs(2 * a)))**2 + self._singM(),
            0
        )
        res += conditional(
            _ufl.And(
                self.x0 <= self.x,
                self.x < self.x0 + 1 / sqrt(abs(2 * a)),
            ),
            -a * (self.x - self.x0 - 1 / sqrt(abs(2 * a)))**2 + self._singP(),
            0
        )
        res += conditional(
            self.x0 + 1 / (_ufl.sign(a) * sqrt(abs(2 * a))) <= self.x,
            self._singP(),
            self._singM()
        )
        return res


def Task_light(x: SpatialCoordinate, data: _pars.Param_light):
    res = Light_collection(
        x=x,
        x0=data.left,
        slope=data.slope,
    ).create(kind=data.kind)
    res *= Light_collection(
        x=x,
        x0=data.right,
        slope=-data.slope,
    ).create(kind=data.kind)
    return res


class BCS_collection:

    def __init__(
        self, indic: Comp_Indicators, const: Task_Consts, data_mesh: _pars.Param_mesh
    ):
        self.indic = indic
        self.const = const
        self.data_mesh = data_mesh

    def create(self, kind, **parametrs):
        assert kind in self._kind_ready(get=True).keys(), 'Not implemented method'
        return getattr(self, kind)(**parametrs)

    @staticmethod
    def _kind_ready(func=None, get=False, l={}):
        if not get:
            args = set(_inspect.getfullargspec(func).args)
            args.remove('self')
            l.update({func.__name__: ','.join(args)})
            return func
        else:
            return l

    @classmethod
    def get_kinds(cls, view=True):
        kinds = cls._kind_ready(get=True)
        if not view:
            return kinds
        else:
            [
                print(kind, ':', args.replace(',', ', ')) for kind,
                args in kinds.items()
            ]

    @_kind_ready
    def close(self, f: _fem.Function):
        return 0

    @_kind_ready
    def robin(self, f: _fem.Function, const, ext):
        return -const * (ext-f)

    @_kind_ready
    def fixed_flux(self, f: _fem.Function, value):
        return -value

    @_kind_ready
    def like_inside(self, f, N_ext, P_ext):
        h = (
            self.data_mesh.right - self.data_mesh.left
        ) / self.data_mesh.domain_intervals
        n, p = self.indic.N, self.indic.P
        if f == n:
            flux_N = 0
            flux_N += -self.const.A_NM * (N_ext-n)
            flux_N += +self.const.A_NM * p * (N_ext-n)
            flux_N += -self.const.E_NP * p * (N_ext-n)
            flux_N += -self.const.A_NM * n * (P_ext-p)
            flux_N += +self.const.E_NP * n * (P_ext-p)
            return flux_N / h
        elif f == p:
            flux_P = 0
            flux_P += -self.const.B_PM * (P_ext-p)
            flux_P += +self.const.B_PM * n * (P_ext-p)
            flux_P += -self.const.E_NP * n * (P_ext-p)
            flux_P += -self.const.B_PM * p * (N_ext-n)
            flux_P += +self.const.E_NP * p * (N_ext-n)
            return flux_P / h


class Comp_bcs(Comp_INTERFACE):

    def action(
        self,
        comp,
        data: _pars.Param_bcs,
        indic: Comp_Indicators,
        data_mesh: _pars.Param_mesh,
        const: Task_Consts
    ):
        factory = BCS_collection(indic=indic, const=const, data_mesh=data_mesh)
        return factory.create(
            kind=data.kind,
            f=indic.get_value(comp),
            **data.get_param(comp),
        )
