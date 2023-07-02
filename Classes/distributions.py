from classes import Base as _b
import ufl as _ufl


class Simple1D:

    def __init__(self, x: _b.SpatialCoordinate, x0, smoothing):
        self.x = x
        self.x0 = x0
        self.smoothing = smoothing

    def create(self, kind: str):
        assert kind in Simple1D._style(get=True), 'Not implemented method'
        return getattr(self, kind)()

    @staticmethod
    def _style(func=None, get=False, l=set()):
        if not get:
            l.add(func.__name__)
            return func
        else:
            return l

    @classmethod
    @property
    def styles(cls):
        return cls._style(get=True)

    def _singP(self):
        return (1 + _ufl.sign(self.smoothing)) / 2

    def _singM(self):
        return (1 - _ufl.sign(self.smoothing)) / 2

    @_style
    def step(self):
        return _b.conditional(
            self.x0 <= self.x,
            self._singP(),
            self._singM(),
        )

    @_style
    def sigmoid(self):
        a = self.smoothing * 5
        return 1 / (1 + _b.exp(-a * (self.x - self.x0)))

    @_style
    def trapstep(self):
        a = self.smoothing
        result = _b.conditional(
            (self.x0 - 1 / abs(2 * a) <= self.x)|_b.ufl_and|
            (self.x < self.x0 + 1 / abs(2 * a)),
            a * (self.x - self.x0) + 0.5,
            0,
        )
        result += _b.conditional(
            self.x0 + 1 / (2*a) <= self.x,
            self._singP(),
            self._singM(),
        )
        return result

    @_style
    def parab(self):
        a = self.smoothing * 5
        result = _b.conditional(
            (self.x0 - 1 / _b.sqrt(abs(2 * a)) <= self.x)|_b.ufl_and|
            (self.x < self.x0),
            a * (self.x - self.x0 + 1 / _b.sqrt(abs(2 * a)))**2 + self._singM(),
            0,
        )
        result += _b.conditional(
            (self.x0 <= self.x)|_b.ufl_and|
            (self.x < self.x0 + 1 / _b.sqrt(abs(2 * a)), ),
            -a * (self.x - self.x0 - 1 / _b.sqrt(abs(2 * a)))**2 +
            self._singP(),
            0,
        )
        result += _b.conditional(
            self.x0 + 1 / (_ufl.sign(a) * _b.sqrt(abs(2 * a))) <= self.x,
            self._singP(),
            self._singM())
        return result

class A:
    g=1
    @property
    def f(self):
        return 1
    def foo(self):
        return 1

a= A()
a.f
a.foo
a.g