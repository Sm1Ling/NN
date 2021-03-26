import random
from Oscillator import *


class ReSyst(object):

    def __init__(self, weights_memory_size, error_memory_size, learning_rate, oscillators_num):
        self.lr = learning_rate
        self.s = weights_memory_size  # как долго запоминать вес
        self.M = error_memory_size  # как долго высчитывать ошибку
        self.O_num = oscillators_num
        self.K = 0.1
        self.Q = 0.2
        self.sqr_v = 0

        self.oscillators = []
        for i in range(self.O_num):
            self.oscillators.append(Oscillator(self.K, self.Q, self.lr))

        self.v = []  # массив эталонных ответов, запоминаются последние M
        self.output = [0 for i in range(self.O_num)]

        self.av_x = []

        self.Error = 0.  # ошибка

    def __update_x(self):
        tmp = 0
        for i in range(self.O_num):
            self.oscillators[i].update_x(self.s)
            tmp += self.oscillators[i].x[0]
        tmp /= self.O_num

        self.av_x.insert(0, tmp)

        if len(self.av_x) > self.s:
            self.av_x.pop()

    def __update_dx(self, input, av_x):

        tmp = 0
        for i in range(self.O_num):
            self.oscillators[i].update_dx(input[i], av_x[0], self.s, self.O_num)

    def __update_f(self):
        for i in range(self.O_num):
            self.oscillators[i].update_f(self.s, self.M)
            self.output[i] = self.oscillators[i].f[0]

    def __update_v(self, y):
        for i in range(self.O_num):
            self.oscillators[i].update_v(y[i], self.M)
        self.v.insert(0, y)  # сохраняет последние М эталонных ответов для каждого осциллятора
        if len(self.v) > self.M:
            self.v.pop()

    def __count_error(self):

        self.sqr_v = 0
        sqr_fv = 0
        for i in range(len(self.v)):
            for j in range(self.O_num):
                self.sqr_v += self.v[i][j] ** 2  # сумма квадратов последниъ эталонных ответов каждого осциллятора

        for i in range(self.O_num):
            sqr_fv += self.oscillators[i].count_error()

        self.Error = (sqr_fv / self.sqr_v)

    def __backpropagate_weights(self):
        for i in range(self.O_num):
            self.oscillators[i].backpropagate_weights(self.s, self.Error)

    def get_new_element(self, sinx, y):

        self.__update_x()
        self.__update_dx(sinx, self.av_x)
        self.__update_f()
        self.__update_v(y)
        self.__count_error()

        if self.Error >= 0.01:
            self.__backpropagate_weights()