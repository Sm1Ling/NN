from MultiLayer import *
from PIL import Image
import numpy as np
import os 

def picture_to_array(path):
    img = Image.open(path)
    img = img.resize((30,30))
    imgarray = np.asarray(img,dtype='uint8')

    bytearr = [ 0 for i in  range(imgarray.shape[0]*imgarray.shape[1])]


    for i in range(imgarray.shape[0]):
        for j in range(imgarray.shape[1]):
            if(imgarray[i][j][0]!=255 or imgarray[i][j][1]!=255 or imgarray[i][j][2]!=255):
                bytearr[i*imgarray.shape[0] + j] = 1 # черный пиксель 
    return bytearr
           
           
def main():
    dirpath = os.getcwd()
    input_dataA = []
    input_dataB = []
    input_data = []
    out_data = []

    for file in os.listdir(dirpath + "\\InputImages\\A"):
        input_dataA.append(picture_to_array(dirpath + "\\InputImages\\A\\" + file)) #получаю все массивы битов для А
    
    for file in os.listdir(dirpath + "\\InputImages\\B"):
        input_dataB.append(picture_to_array(dirpath + "\\InputImages\\B\\" + file)) #получаю все массивы битов для B

    for i in range(len(input_dataA)):
        input_data.append(input_dataA[i])
        input_data.append(input_dataB[i])
        out_data.append(1)
        out_data.append(0)


    m_layer = MultiLayer(4, len(input_data[0]), len(input_data[0]),1)
    m_layer._train(input_data,out_data)


    string = "line"
    while(string != "Exit"):
        print("Write Exit ro finish programm, or name of file to test the neuron web")

        string = input()
        if(os.path.isfile(dirpath + "\\InputImages\\" + string)):
            m_layer._complete_count(picture_to_array(dirpath + "\\InputImages\\" + string))
            print("Output is ", *m_layer._output)
            print("This is probably A" if m_layer._output[0] > 0.5  else "This is probably B")
        if(string == "Save"):
            print("Saving")
        else:
            continue
    

if __name__ == '__main__':
    main()
