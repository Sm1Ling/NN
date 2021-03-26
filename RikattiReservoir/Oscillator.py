import random


class Oscillator(object):

    def __init__(self, K, Q, LR):
        self.f = []  # массив моих ответов, запоминаются последние M
        self.dx = []  # массив значений х' во времени, запоминаются последние s
        self.dy = 0
        self.dz = 0

        self.x = []  # последнее значение х во времени, запоминаются последние s
        self.y = []  # последнее значение y во времени, запоминаются последние s
        self.z = []  # последнее значение z, запоминаются последние s

        self.K = K
        self.Q = Q
        self.lr = LR

        self.dAdwx = [[], [], []]  # массив производных x y z по Wx
        self.dxdwx = []  # массив производных x'  Wx
        self.dydwx = 0  # dy'(t)/dWx
        self.dzdwx = 0  # dz'(t)/dWx

        self.w = []  # массив весов памяти, обновляются последние s
        self.v = []
        self.input = []

        self.wx = random.uniform(0.11, 0.2) - 0.1  # вес при уравнениях dx dy dz

    def update_x(self, s):
        if len(self.x) == 0:

            self.dAdwx[0].append(0)  # dx(0)/dWx
            self.dAdwx[1].append(0)  # dy(0)/dWx
            self.dAdwx[2].append(0)  # dz(0)/dWx

            self.x.append(0)  # x(0)
            self.y.append(0)  # y(0)
            self.z.append(0)  # z(0)


        else:
            # использует данные из t-1
            self.dAdwx[0].insert(0, self.dAdwx[0][0] + self.dxdwx[0])  # dx(t)/dWx = dx(t-1)/dWx + dx'(t-1)dWx
            self.dAdwx[1].insert(0, self.dAdwx[1][0] + self.dydwx)  # dy(t)/dWx = dy(t-1)/dWx + dy'(t-1)dWx
            self.dAdwx[2].insert(0, self.dAdwx[2][0] + self.dzdwx)  # dz(t)/dWx = dz(t-1)/dWx + dz'(t-1)dWx
            # использует данные из t-1
            self.x.insert(0, self.x[0] + self.dx[0])  # x(t) = x(t-1) + x'(t-1)
            self.y.insert(0, self.y[0] + self.dy)
            self.z.insert(0, self.z[0] + self.dz)

        if len(self.x) > s:
            self.x.pop()
            self.y.pop()
            self.z.pop()

            self.dAdwx[0].pop()
            self.dAdwx[1].pop()
            self.dAdwx[2].pop()

    def update_dx(self, input, av_x, s, n):
        if len(self.dx) == 0:

            self.dxdwx.append(0)  # dx'(0)/dWx
            self.dydwx = 0  # dy'(0)/dWx
            self.dzdwx = 0  # dz'(0)/dWx

            self.dx.append(input)  # x'(0)
            self.dy = 0
            self.dz = 0

        else:
            # использует данные из t
            self.dx.insert(0, self.wx * (10 * (self.y[0] - self.x[0]) +
                                         self.K * (self.Q * av_x - self.x[0])) + input)
            # x'(t) = wx*(10(y(t) - x(t) + K*(Q*av_x - x(t))) + sin
            self.dy = self.wx * (-self.x[0] * self.z[0] + 0.5 * self.x[0] - self.y[0])
            # y'(t) = wx*(-x(t)*z(t) + 0.2x(t) - y(t)
            self.dz = self.wx * (self.x[0] * self.y[0] - 8 * self.z[0] / 3)
            # z'(t) = wx*(x(t)*y(t) - 8*z(t)/3)

            # использует данные из t
            # dx'(t)/dWx = 10*(y(t) - x(t)) + K(Q*av_x - x(t)) + wx*(dy(t)/dWx - dx(t)/dWx * (KQ/n - K - 10))
            self.dxdwx.insert(0, 10 * (self.y[0] - self.x[0]) + self.K * (self.Q * av_x - self.x[0])
                              + self.wx * (10 * self.dAdwx[1][0] + self.dAdwx[0][0] *
                                           (self.Q * self.K / n - self.K - 10)))
            # dy'(t)/dWx = (-x(t)*z(t) + 0.2x(t) - y(t)) +
            # wx*(-x(t)*dz(t)/dWx - dx(t)/dWx*z(t) + 0.2dx(t)/dWx - dy(t)/dWx
            self.dydwx = -self.x[0] * self.z[0] + 0.5 * self.x[0] - self.y[0]
            + self.wx * (-self.x[0] * self.dAdwx[2][0] - self.dAdwx[0][0] * self.z[0]
                         + 0.2 * self.dAdwx[0][0] - self.dAdwx[1][0])
            # dz'(t)/dWx = (x(t)*y(t) - 8*z(t)/3) + wx*( dx(t)/dWx*y(t) + x(t)*dy(t)/dWx - 8*dz(t)/dWx / 3))
            self.dzdwx = self.x[0] * self.y[0] - 8 * self.z[0] / 3 + self.wx * (
                    self.dAdwx[0][0] * self.y[0] + self.x[0] * self.dAdwx[1][0] - 8 * self.dAdwx[2][0] / 3)

        self.input.insert(0, input)

        if len(self.dx) > s:
            self.dx.pop()
            self.input.pop()
            self.dxdwx.pop()

    def update_f(self, s, m):
        if len(self.w) < s:
            self.w.insert(0, random.uniform(0, 1) - 0.5)  # если весов недостаточно -- создаю их
        f_t = 0

        for i in range(len(self.w)):
            f_t += self.dx[i] * self.w[i]  # от x'(t) до x'(t-s)

        self.f.insert(0, f_t)

        if len(self.f) > m:
            self.f.pop()  # забываю старые показатели функции

    def update_v(self, y, m):
        self.v.insert(0, y)

        if len(self.v) > m:
            self.v.pop()  # забываю старые эталонные значения

    def count_error(self):
        sqr_fv = 0
        for i in range(len(self.v)):
            sqr_fv += (self.v[i] - self.f[i]) ** 2
        return sqr_fv

    def backpropagate_weights(self, s, error):
        sqr_v = 0
        for i in range(len(self.v)):  # считаю сумму квадратов эталонных весов, это знаменатель в dE/df
            sqr_v += self.v[i] ** 2

       # delta_wx = 0
       # # корректирую w при формулах х y z.  dE/df * df/dx' * dx'/dwx (учитываю все моменты времени)
       # for i in range(len(self.w)):  # d(w(t)*x'(t) + ... w(t-s)*x'(t-s) )/ dWx
       #     delta_wx += self.w[i] * self.dxdwx[i] / (len(self.w))
       # # delta_wx = self.w[0]*self.dxdwx[0]

        coef = 7
        if len(self.w) == s:
            if 0.5 <= error < 1:
                coef = 5
            elif error < 0.5:
                coef = 3
            elif error > 1.5:
                coef = 1

        #self.wx -= coef * self.lr * 2 * (self.f[0] - self.v[0]) * delta_wx / sqr_v

        for i in range(len(self.w)):
            delta_w = 2 * (self.f[0] - self.v[0]) * self.dx[i]  # dE/df(t) * df(t)/dw
            self.w[i] -= coef  * delta_w / sqr_v  # корректирую все веса памяти
            if self.w[i] < 0.0001:
                self.w[i] = 0
