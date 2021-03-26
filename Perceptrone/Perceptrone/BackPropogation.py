from Layer import *
from Neuron import *

class BackPropogation(object):

    def __init__(self, layers,educ_speed, normal_error, momentum = 0 ):
        self.EDUCATION_SPEED = educ_speed
        self.MOMENTUM = momentum
        self.NORMAL_ERROR = normal_error

        self._error = 0
        self._epoch = 0

        self._layers = layers

        self._deltas = [0 for i in range(len(self._layers))] # для каждого слоя будут дельты
        self._pr_biases = [0 for i in range(len(self._layers))] # для каждого слоя будут запоминаться гиперпараметры
        self._pr_weights = [0 for i in range(len(self._layers))] # для каждого слоя будут запоминаться веса 

        for i in range(len(self._layers)):
            self._deltas[i] = [0 for i in range(len(self._layers[i]._neurons_arr))] # для каждого нейрона в слое будут дельты
            self._pr_biases[i] = [0 for i in range(len(self._layers[i]._neurons_arr))] # для каждого нейрона в слое будут гиперпараметры
            self._pr_weights[i] = [0 for i in range(len(self._layers[i]._neurons_arr))] # для каждого нейрона в слое будут веса

            for j in range(len(self._layers[i]._neurons_arr)):#пройдемся по нейронам в слое
                self._pr_weights[i][j] = [0]*len(self._layers[i]._neurons_arr[j]._weights) # Для каждого нейрона в слое будет массив весов. Изначально ноль

    def training (self, real_out, expected_out):
        error = 0

        for p in range(len(self._layers)-1,0,-1): # по каждому слою начиная с выходного
            if( p == len(self._layers) - 1):
                for k in range(len(self._layers[p]._neurons_arr)): #по каждому нейрону
                     if(isinstance(expected_out,list)):
                        self._deltas[p][k] = (expected_out[k] - real_out[k]) # delta = (exp_out - real_out)*(F'(s))  F' = y * (1-y)
                     else:
                        self._deltas[p][k] = (expected_out - real_out[0])
                     error += self._deltas[p][k]**2     # сначала посчитал в дельте только разность, чтобы потом добавить ее в ошибку 
                     self._deltas[p][k] *= (self._layers[p]._neurons_arr[k]._y*(1 - self._layers[p]._neurons_arr[k]._y))
            else:
                for k in range(len(self._layers[p]._neurons_arr)): #для каждого нейрона в слое
                    deltasum = 0
                    for d in range(len(self._layers[p+1]._neurons_arr)):  # в ряду p беру нейрон k и доавбляю к сумме дельт от нейрона d из слоя p+1 вес номер k( от которого идет) * дельту нейрона d
                        deltasum+= self._deltas[p+1][d]*self._layers[p+1]._neurons_arr[d]._weights[k] #суммирую дельты с верхних слоев * на веса нейронов с тех слоев (Delta = Sum( w_up*delta_up)*F'(s))
                    deltasum *= (self._layers[p]._neurons_arr[k]._y*(1 - self._layers[p]._neurons_arr[k]._y)) #домножаю на производную функции
                    self._deltas[p][k] = deltasum

        self._error += error/2

        for p in range(1,len(self._layers)): 
           for k in range(len(self._layers[p]._neurons_arr)):
               for w in range(len(self._layers[p]._neurons_arr[k]._weights)): # пересчитываю веса. w(t+1) = w(t) + education_speed*delta*(y of prev neuron) + momentum* prev_delta_w
                   
                   deltaw = self.EDUCATION_SPEED * self._deltas[p][k] * self._layers[p-1]._neurons_arr[w]._y
                   self._layers[p]._neurons_arr[k]._weights[w] += deltaw + self.MOMENTUM*self._pr_weights[p][k][w]
                   self._pr_weights[p][k][w] = deltaw
        # на bias обработки у меня нет




