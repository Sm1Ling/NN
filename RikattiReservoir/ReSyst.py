import random
from Oscillator import *


class ReSyst(object):

    def __init__(self, weights_memory_size, error_memory_size, learning_rate, oscillators_num):
        self.lr = learning_rate
        self.s = weights_memory_size  # как долго запоминать вес
        self.M = error_memory_size  # как долго высчитывать ошибку
        self.O_num = oscillators_num
        # K = 0.05  Q = 0.7
        self.sqr_v = 0

        self.oscillators = []
        for i in range(self.O_num):
            self.oscillators.append(RikattiOscillator(self.s, self.M, self.lr))

        self.v = CyclicQueue(self.M)  # массив эталонных ответов, запоминаются последние M
        self.output = [0 for i in range(self.O_num)]

        self.av_x = CyclicQueue(self.s)

        self.Error = 0.  # ошибка

        self.blocked = False

    def __update_x(self,input):
        tmp = 0
        for i in range(self.O_num):
            self.oscillators[i].update_inners_before(input[i], self.av_x.get_last(), self.O_num)
            tmp += self.oscillators[i].x.get_last()
        tmp /= self.O_num

        self.av_x.add_new(tmp)

    def __update_dx(self, input, av_x):
        for i in range(self.O_num):
            self.oscillators[i].update_inners_after(input[i], av_x.get_last(), self.O_num)

    def __update_f(self):
        for i in range(self.O_num):
            self.oscillators[i].update_f()
            self.output[i] = self.oscillators[i].f.get_last()

    def __update_v(self, y):
        for i in range(self.O_num):
            self.oscillators[i].update_v(y[i])
        self.v.add_new(y)  # сохраняет последние М эталонных ответов для каждого осциллятора

    def __count_error(self):

        self.sqr_v = 0
        sqr_fv = 0
        for i in range(self.v.length()):
            for j in range(self.O_num):
                self.sqr_v += self.v.get(i)[j] ** 2  # сумма квадратов последниъ эталонных ответов каждого осциллятора

        for i in range(self.O_num):
            sqr_fv += self.oscillators[i].count_error()

        self.Error = (sqr_fv / self.sqr_v)

    def __backpropagate_weights(self):
        for i in range(self.O_num):
            self.oscillators[i].backpropagation()

    def get_new_element(self, inp, y):

        self.__update_x(inp)
        self.__update_dx(inp, self.av_x)
        self.__update_f()
        self.__update_v(y)
        self.__count_error()

        if not self.blocked:
           # if(self.Error < 0.003):
                #self.blocked = True
            self.__backpropagate_weights()
