import math
import random
from abc import ABC, abstractmethod


class Oscillator(ABC):
    def __init__(self, S, M, LR):
        self.f = CyclicQueue(M)  # массив подсчитаных выводных значений
        self.s = S  # количество весов памяти
        self.m = M  # количество моментов времени расчета ошибки
        self.lr = LR  # скорость обучеия
        self.w = []  # массив весов памяти, обновляются последние s
        self.v = CyclicQueue(M)  # массив эталонных выводных значений
        self.input = CyclicQueue(S)  # массив вводных данных
        self.wx = random.uniform(0.16, 0.2) - 0.1  # вес при уравнениях dx dy dz
        self.preFParam = CyclicQueue(S)  # массив параметра для использования в f
        self.x = CyclicQueue(S)

    @abstractmethod
    def update_inners_before(self, input, averageParam, numOfOscillators):
        pass

    @abstractmethod
    def update_inners_after(self, input, averageParam, numOfOscillators):
        pass

    def update_f(self):
        if len(self.w) < self.s:  # если весов недостаточно -- создаю их
            if len(self.w) != 0 and self.w[-1] > 0:  # равномерно создаю отрицательные и положительные веса
                self.w.append(random.uniform(0, 0.5) - 0.5)
            else:
                self.w.append(random.uniform(0.5, 1) - 0.5)
        f_t = 0

        for i in range(len(self.w)):
            f_t += self.preFParam.get(i) * self.w[i]  # от x'(t) до x'(t-s)
        self.f.add_new(f_t)

    def update_v(self, y):
        self.v.add_new(y)

    def count_error(self):
        sqr_fv = 0
        for i in range(self.v.length()):
            sqr_fv += (self.v.get(i) - self.f.get(i)) ** 2
        return sqr_fv

    @abstractmethod
    def backpropagation(self):
        pass


class LorensOscillator(Oscillator):

    def __init__(self, S, M, LR, K, Q, trainWx):

        super().__init__(S, M, LR)
        #  dx инициализирован в базовом классе
        self.dy = 0
        self.dz = 0

        self.y = CyclicQueue(S)  # последнее значение y во времени, запоминаются последние s
        self.z = CyclicQueue(S)  # последнее значение z, запоминаются последние s

        self.K = K
        self.Q = Q

        self.dAdwx = [CyclicQueue(S), CyclicQueue(S), CyclicQueue(S)]  # массив производных x y z по Wx
        self.dxdwx = CyclicQueue(S)  # массив производных x'  Wx
        self.dydwx = 0  # dy'(t)/dWx
        self.dzdwx = 0  # dz'(t)/dWx

        self.isWxTrained = trainWx

    def update_inners_before(self, input, averageParam, numOfOscillators):
        if self.x.length() == 0:

            self.dAdwx[0].add_new(0)  # dx(0)/dWx
            self.dAdwx[1].add_new(0)  # dy(0)/dWx
            self.dAdwx[2].add_new(0)  # dz(0)/dWx

            self.x.add_new(0)  # x(0)
            self.y.add_new(0)  # y(0)
            self.z.add_new(0)  # z(0)


        else:
            # использует данные из t-1
            if self.isWxTrained:
                self.dAdwx[0].add_new(
                    self.dAdwx[0].get_last() + self.dxdwx.get_last())  # dx(t)/dWx = dx(t-1)/dWx + dx'(t-1)dWx
                self.dAdwx[1].add_new(self.dAdwx[1].get_last() + self.dydwx)  # dy(t)/dWx = dy(t-1)/dWx + dy'(t-1)dWx
                self.dAdwx[2].add_new(self.dAdwx[2].get_last() + self.dzdwx)  # dz(t)/dWx = dz(t-1)/dWx + dz'(t-1)dWx
            # использует данные из t-1
            self.x.add_new(self.x.get_last() + self.preFParam.get_last())  # x(t) = x(t-1) + x'(t-1)
            self.y.add_new(self.y.get_last() + self.dy)
            self.z.add_new(self.z.get_last() + self.dz)

    def update_inners_after(self, input, averageParam, numOfOscillators):
        if self.preFParam.length() == 0:

            self.dxdwx.add_new(0)  # dx'(0)/dWx
            self.dydwx = 0  # dy'(0)/dWx
            self.dzdwx = 0  # dz'(0)/dWx

            self.preFParam.add_new(input)  # x'(0)
            self.dy = 0
            self.dz = 0

        else:
            # использует данные из t
            self.preFParam.add_new(self.wx * (10 * (self.y.get_last() - self.x.get_last()) +
                                              self.K * (self.Q * averageParam - self.x.get_last())) + input)
            # x'(t) = wx*(10(y(t) - x(t) + K*(Q*av_x - x(t))) + sin
            self.dy = self.wx * (-self.x.get_last() * self.z.get_last() + 0.5 * self.x.get_last() - self.y.get_last())
            # y'(t) = wx*(-x(t)*z(t) + 0.2x(t) - y(t)
            self.dz = self.wx * (self.x.get_last() * self.y.get_last() - 8 * self.z.get_last() / 3)
            # z'(t) = wx*(x(t)*y(t) - 8*z(t)/3)

            # использует данные из t
            if self.isWxTrained:
                # dx'(t)/dWx = 10*(y(t) - x(t)) + K(Q*av_x - x(t)) + wx*(dy(t)/dWx - dx(t)/dWx * (KQ/n - K - 10))
                self.dxdwx.add_new(
                    10 * (self.y.get_last() - self.x.get_last()) + self.K * (self.Q * averageParam - self.x.get_last())
                    + self.wx * (10 * self.dAdwx[1].get_last() + self.dAdwx[0].get_last() *
                                 (self.Q * self.K / numOfOscillators - self.K - 10)))
                # dy'(t)/dWx = (-x(t)*z(t) + 0.2x(t) - y(t)) +
                # wx*(-x(t)*dz(t)/dWx - dx(t)/dWx*z(t) + 0.2dx(t)/dWx - dy(t)/dWx
                self.dydwx = -self.x.get_last() * self.z.get_last() + 0.5 * self.x.get_last() - self.y.get_last()
                + self.wx * (
                        -self.x.get_last() * self.dAdwx[2].get_last() - self.dAdwx[0].get_last() * self.z.get_last()
                        + 0.2 * self.dAdwx[0].get_last() - self.dAdwx[1].get_last())
                # dz'(t)/dWx = (x(t)*y(t) - 8*z(t)/3) + wx*( dx(t)/dWx*y(t) + x(t)*dy(t)/dWx - 8*dz(t)/dWx / 3))
                self.dzdwx = self.x.get_last() * self.y.get_last() - 8 * self.z.get_last() / 3 + self.wx * (
                        self.dAdwx[0].get_last() * self.y.get_last() + self.x.get_last() * self.dAdwx[1].get_last()
                        - 8 * self.dAdwx[2].get_last() / 3)

        self.input.add_new(input)

    def backpropagation(self):
        sqr_v = 0
        for i in range(self.v.length()):  # считаю сумму квадратов эталонных весов, это знаменатель в dE/df
            sqr_v += self.v.get(i) ** 2

        delta_wx = 0

        coef = 0.1
        if (len(self.w) > 30):
            coef = 0.1

        if self.isWxTrained:
            # корректирую w при формулах х y z.  dE/df * df/dx' * dx'/dwx (учитываю все моменты времени)
            # delta_wx = self.w[-1]*self.dxdwx.get_last()
            for i in range(len(self.w)):  # d(w(t)*x'(t) + ... w(t-s)*x'(t-s) )/ dWx
                delta_wx += self.w[i] * self.dxdwx.get(i) / (len(self.w))  # от t-s к t идет
            self.wx -= coef * self.lr * 2 * (self.f.get_last() - self.v.get_last()) * delta_wx / sqr_v

        for i in range(len(self.w)):
            delta_w = 2 * (self.f.get_last() - self.v.get_last()) * self.preFParam.get(i)  # dE/df(t) * df(t)/dw
            self.w[i] -= coef * self.lr * delta_w / sqr_v  # корректирую все веса памяти от t-s до t
            if abs(self.w[i]) < 0.0001:
                self.w[i] = 0

    def __str__(self):
        # ERROR -- relies on ReSyst
        # DENOMINATOR
        # AVERAGE
        line = 'wx = %.3f' % self.wx
        line += '\ndx = %.3f' % self.preFParam.get_last()
        line += '\ndy = %.3f' % self.dy
        line += '\ndz = %.3f' % self.dz
        line += '\ndx\'(t)/dWx = %.3f' % self.dxdwx.get_last()
        line += '\ndy\'(t)/dWx = %.3f' % self.dydwx
        line += '\ndz\'(t)/dWx = %.3f' % self.dzdwx
        line += '\nf = %.3f' % self.f.get_last()
        line += '\ndx(t)/dWx = %.3f' % self.dAdwx[0].get_last()
        line += '\ndy(t)/dWx = %.3f' % self.dAdwx[1].get_last()
        line += '\ndy(t)/dWx = %.3f' % self.dAdwx[2].get_last()
        line += '\nx = %.3f' % self.x.get_last()
        line += '\ny = %.3f' % self.y.get_last()
        line += '\nz = %.3f' % self.z.get_last()
        return line


class RikattiOscillator(Oscillator):

    def __init__(self, S, M, LR):
        super().__init__(S, M, LR)
        self.b = 1
        self.c = 0.3
        self.y = CyclicQueue(S)

    def update_inners_before(self, input, averageParam, numOfOscillators):
        self.input.add_new(input)
        sqrt = math.sqrt(self.input.get_last())
        self.y.add_new(math.sin(sqrt + math.cos(self.input.get_last())))
        self.x.add_new(
            math.cos(sqrt + math.cos(self.input.get_last())) * (0.5 / sqrt - math.sin(self.input.get_last())))  # dy/dt
        self.preFParam.add_new(
            5 * self.x.get_last() - 5 * self.b * self.y.get_last() - 25 * self.c * (self.y.get_last() ** 2))

    def update_inners_after(self, input, averageParam, numOfOscillators):
        pass

    def backpropagation(self):
        sqr_v = 0
        for i in range(self.v.length()):  # считаю сумму квадратов эталонных весов, это знаменатель в dE/df
            sqr_v += self.v.get(i) ** 2

        coef = 1

        for i in range(len(self.w)):
            delta_w = 2 * (self.f.get_last() - self.v.get_last()) * self.preFParam.get(i)  # dE/df(t) * df(t)/dw
            self.w[i] -= coef * self.lr * delta_w / sqr_v  # корректирую все веса памяти от t-s до t

    def __str__(self):
        # ERROR -- relies on ReSyst
        # DENOMINATOR
        # AVERAGE
        line = 'wx = %.3f' % self.wx
        line += '\na(t) = %.3f' % self.preFParam.get_last()
        line += '\nf = %.3f' % self.f.get_last()
        line += '\ndy(t)/dt = %.3f' % self.x.get_last()
        line += '\ny(t) = %.3f' % self.y.get_last()
        return line


class CyclicQueue(object):

    def __init__(self, msize):
        self.__left = 0
        self.__m_size = msize
        self.__size = 0
        self.__arr = [0 for i in range(msize)]

    def add_new(self, element):
        if self.__size == self.__m_size:
            self.__increase(element)
        else:
            self.__arr[self.__size] = element
            self.__size += 1

    def length(self):
        return self.__size

    def get(self, index):
        if self.__size != 0:
            return self.__arr[(self.__left + index) % self.__size]
        else:
            return 0

    def get_first(self):
        if self.__size != 0:
            return self.__arr[self.__left]
        else:
            return 0

    def get_last(self):
        return self.get(self.__size - 1)

    def __increase(self, element):
        self.__arr[self.__left] = element
        self.__left = (self.__left + 1) % self.__m_size
