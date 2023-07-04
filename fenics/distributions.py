from . import operators as _fn
import ufl as _ufl


class Simple1D:
    """
    Container class for different function distributions in 1D
    """
    def __init__(self, x: _fn.SpatialCoordinate, x0, smoothing):
        self.x = x
        self.x0 = x0
        self.smoothing = smoothing

    def create(self, style: str):
        assert style in Simple1D._style(get=True), 'Not implemented method'
        return getattr(self, style)()

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
    def stepwise(self):
        return _fn.conditional(
            self.x0 <= self.x,
            self._singP(),
            self._singM(),
        )

    @_style
    def sigmoid(self):
        a = self.smoothing * 5
        return 1 / (1 + _fn.exp(-a * (self.x - self.x0)))

    @_style
    def trapstep(self):
        a = self.smoothing
        result = _fn.conditional(
            (self.x0 - 1 / abs(2 * a) <= self.x)|_fn.And|
            (self.x < self.x0 + 1 / abs(2 * a)),
            a * (self.x - self.x0) + 0.5,
            0,
        )
        result += _fn.conditional(
            self.x0 + 1 / (2*a) <= self.x,
            self._singP(),
            self._singM(),
        )
        return result

    @_style
    def parab(self):
        a = self.smoothing * 5
        result = _fn.conditional(
            (self.x0 - 1 / _fn.sqrt(abs(2 * a)) <= self.x)|_fn.And|
            (self.x < self.x0),
            a * (self.x - self.x0 + 1 / _fn.sqrt(abs(2 * a)))**2 +
            self._singM(),
            0,
        )
        result += _fn.conditional(
            (self.x0 <= self.x)|_fn.And|
            (self.x < self.x0 + 1 / _fn.sqrt(abs(2 * a)), ),
            -a * (self.x - self.x0 - 1 / _fn.sqrt(abs(2 * a)))**2 +
            self._singP(),
            0,
        )
        result += _fn.conditional(
            self.x0 + 1 / (_ufl.sign(a) * _fn.sqrt(abs(2 * a))) <= self.x,
            self._singP(),
            self._singM())
        return result
