import math
from abc import ABC, abstractmethod
from sympy import Matrix, zeros, eye


class BaseFunc(ABC):

    @abstractmethod
    def input_func(self, x):
        pass

    @abstractmethod
    def output_func(self):
        pass


class RikattiFunc(BaseFunc):
    """
    Riccati Equation  y'(t) = a(t) + b(t) * y(t) + c(t) * (y(t))^2
    y(t) = 5 * sin(sqr(x) + cos(x))
    y'(t) = 5 * cos( sqr(x) + cos(x)) * ( 0.5/sqr(x) - sin(x))
    a(t) = 5 * ( cos(sqr(x) + cos(x)) * ( 0.5/sqr(x) - sin(x))) -  b(t) * sin(sqr(x) + cos(x)) -
    - 5 * c(t) * (sin(sqr(x) + cos(x)))^2)
    """

    def __init__(self, b, c):
        self._b = b
        self._c = c
        self._y = 0
        self._x = 0
        self.__sqrt = 0

    def input_func(self, x):
        self.__sqrt = math.sqrt(x)
        self._x = x
        self._y = 5 * math.sin(self.__sqrt + math.cos(self._x))
        return self._x

    def output_func(self):
        return 5 * math.cos(self.__sqrt + math.cos(self._x)) * (
                (1 / (2 * self.__sqrt)) - math.sin(self._x)) - self._b * self._y - self._c * (self._y ** 2)


class SinCosFunc(BaseFunc):
    """
    Sinus to Cosinus transformation
    """

    def __init__(self):
        self._x = 0

    def input_func(self, x):
        self._x = x
        return math.sin(self._x)

    def output_func(self):
        return math.cos(self._x)


class ThermalFunc(BaseFunc):
    """
    R_out = 0.2527
    R_in = 0.2121
    ρ = 7.95(1 + 1.7 * 10-5T)
    с = 0.527135 + 9.56*10-5(T – 273) + 7.425*10-10(T-743)3
    q(t) = 302.87
    W = 735
    k = 0.146(0.64 + 9.65 * 10-4T)
    h(0) = 15
    h(t) = 15 – 3t  при t<1  и  h = 2 при t > 1
    Tc  = 679 + 17t   если t< 1   и  696 при t>1
    T_0 = 57(r – Rin)/(Rout – Rin) + 753
    """

    def __init__(self, r):
        self._dx = 0
        self._r = r
        self._x = 0
        self._Rin = 0.2121
        self._Rout = 0.2527
        self._T = 57 * (self._r - self._Rin) / (self._Rout - self._Rin) + 753
        self._W = 735
        self._D = zeros(2)
        self._d = zeros(2, 1)
        self._g0 = Matrix([745.875, 799.3125])  # initial values of g(0) are determined by
        self._g = self._g0
        # ensuring that the trial function calculated at
        # t = 0, fits best the initial value
        self._q = 302.87
        self._f1 = ((self._Rout - self._r) / (self._Rout - self._Rin)) ** 2
        self._f2 = 1 - self._f1

    def input_func(self, x):

        if x == 0:
            self._x = 0  # это время
            return 0
        self._dx = x - self._x
        self._x = x
        c = self.c()
        p = self.p()
        cp = c * p
        k = self.k()
        h = self.h(self._x)
        self._D = Matrix([[-6075.1493053 * k / cp, 3241.0149873 * k / cp],
                          [(6075.1492988 * k + 40.266966 * h) / cp, (-3241.0149769 * k + 62.5517792 * h) / cp]])

        self._d = Matrix([(-self._W + 127.9671763 * self._q - 40.266966 * self.Tc(self._x) * h) / cp,
                          (-self._W - 33.7974811 * self._q + 62.5517792 * self.Tc(self._x) * h) / cp])
        return self._x

    def output_func(self):
        if self._x == 0:
            return self._T

        exp = (-self._D*self._dx).exp()
        res = (eye(2) - exp) * (self._D.inv(method="LU")) * self._d + exp*self._g
        self._g = res
        self._T = self._g[0, 0] * self._f1 + self._g[1, 0]*self._f2
        return self._T

    def f1(self):
        return ((self._Rout - self._r) / (self._Rout - self._Rin)) ** 2

    def p(self):
        return 7.95 * (1 + 1.7 * self._T / 100000)
        #return 7.95 * 1.17

    def k(self):
        return 0.146 * (0.64 + 9.65 * self._T / 10000)
        #return 0.146 * 0.64 + 0.965

    def h(self, t):
        if t < 1:
            return 15 - 3 * t
        else:
            return 2

    def Tc(self, t):
        if t < 1:
            return 679 + 17 * t
        else:
            return 696

    def c(self):
        return 0.527135 + 9.56 * (self._T - 273) / 100000 + (7.425 * (self._T - 743) ** 3) / 10000000000
        #return 0.52814
