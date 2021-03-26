from Layer import *
from Neuron import *
from BackPropogation import *

class MultiLayer():
    def __init__(self, layercount, inputcount,hidencount, outputcount):

        self._layers = [0 for i in range(layercount)]
        self._output = [0 for i in range(outputcount)]
        self._input = [0 for i in range(inputcount)]
        self._regenerate_layers(layercount,hidencount)
        self._error = 0
        
       

    def _regenerate_layers(self, layercount,hidencount):
        self._layers = [0 for i in range(layercount)]
        for i in range(layercount):
           if(i == 0):
               self._layers[i] = Layer(len(self._input),0,0)
           elif(i!= layercount-1):
               self._layers[i] = Layer(hidencount,len(self._layers[i-1]._neurons_arr),1)
           else:
               self._layers[i] = Layer(len(self._output),len(self._layers[i-1]._neurons_arr),2)

    def _complete_count(self, input_data):
        self._input = input_data

        for i in range(len(self._layers[0]._neurons_arr)):
            self._layers[0]._neurons_arr[i]._y = input_data[i]
            self._layers[0]._out_info_data[i] = input_data[i]

        for i in range(1,len(self._layers)):
            self._layers[i]._get_inner_info(self._layers[i-1]._out_info_data)
            self._layers[i]._set_out_info()
        self._output = self._layers[len(self._layers)-1]._out_info_data

    def _train(self, all_the_inputs, expected_outputs):
        back_prop_handler = BackPropogation(self._layers, 0.1, 0.005)

        print("Education Started! Parameters: Speed", back_prop_handler.EDUCATION_SPEED, "  Normal Error", back_prop_handler.NORMAL_ERROR, 
             "  Momentum", back_prop_handler.MOMENTUM)

        doWhileHelper = 1

        while(doWhileHelper == 1 or self._error > back_prop_handler.NORMAL_ERROR):
            doWhileHelper = 0

            self._error = 0
            back_prop_handler._epoch +=1
            for i in range(len(all_the_inputs)):
                self._complete_count(all_the_inputs[i]) # считаю что дает мне обработка входных данных
                back_prop_handler.training(self._output,expected_outputs[i])
                self._error += back_prop_handler._error
                print("Epoch ",back_prop_handler._epoch,"  Training Error: ",back_prop_handler._error, "  Output: ",self._output, "Expected: ", expected_outputs[i])

                back_prop_handler._error = 0
            print("Total Epoch's error is ", self._error)

        print("Training has finished with error = ", self._error)



            




