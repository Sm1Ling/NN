import numpy as np
import random


class Neuron(object):
    

      def __init__(self,weight_count, bias = 0, effective_input = 0):
          self._s = effective_input
          self._b = bias #нормально не разобрано
          self._y = 0 
          self.__random_wieghts(weight_count) #создаю массив весов в соответствии с указанным количеством ветвлений

      def __activation_function(self, param):
          return 1/(1+np.e**(-param))

      def __str__(self):
          return "s: " + str(self.__s) + "  w: " + str(self.__w) + "  o: " + str(self.__o) + "  y: " + str(self.__y)

      def __random_wieghts(self, weight_count):
          self._weights = [0 for i in range(weight_count)] #входящие веса в данный нейрон
          for i in range(len(self._weights)):
              self._weights[i] = random.uniform(0,1)-0.5

      def _out_y(self):
         self._y = self.__activation_function(self._s)
         return self._y


      def _calculate_s(self, ins):
        temp = 0
        for i in range(len(ins)):
            temp += ins[i]*self._weights[i]
        temp += self._b
        self._s = temp
             
             


     