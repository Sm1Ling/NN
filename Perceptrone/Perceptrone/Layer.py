from Neuron import *

class Layer(object):
   
    def __init__(self, neurons_count, weights_count, layertype = 1): #layer type = 0 for initial, 1 for hiden , 2 for out
        self._layer_type = layertype
        self._out_info_data = [0 for i in range(neurons_count)]
        self._neurons_arr = [0 for i in range(neurons_count)]
        for i in range(neurons_count):
            self._neurons_arr[i] = Neuron(weights_count) #прикол в bias не учитывается

    def _set_out_info(self):
        """Fill the array of out data with each neuron _y"""
        for i in range(len(self._neurons_arr)):
            self._out_info_data[i] = self._neurons_arr[i]._out_y()

    def _get_inner_info(self, inner_info_data):
        """Make each neuron to get the inner_info_data from previous layer"""
        for i in range(len(self._neurons_arr)):
            self._neurons_arr[i]. _calculate_s(inner_info_data)
