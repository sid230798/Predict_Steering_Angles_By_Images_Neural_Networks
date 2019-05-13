'''

Name :- Siddharth Nahar
Entry No :- 2016csb1043
Date :- 20/10/18

Purpose :- 

	1. Main function which centralize all Operations
	2. Start Training of Functions
'''

from PreProcess import preProcess
from layerStructure import layer
from Training import run
import numpy as np
import sys

print('Started Preprocessing of Data ........')

TrainSetX,TrainSetY,TestSetX,TestSetY = preProcess(sys.argv[1])

print('PreProcessing Ended')

n = input('Number of Hidden Layers in input : ')
func = input('Enter the name of Activation Function as sigmoid,tanh, or relu : ')
LayerList = list()
DropOut = list()
n = int(n)
prev = 1024

for i in range(0,n+1):

	dropPer = input('Enter the Dropout fraction for '+str(i+1)+' Layer : ')
	dropPer = float(dropPer)
	DropOut.append(1-dropPer)

for i in range(0,n):

	new = input('Enter number of Nodes in '+str(i+1)+' Hidden layer : ')
	new = int(new)
	Obj = layer(prev,new,activation = func,keyProb = DropOut[i+1])
	LayerList.append(Obj)
	prev = new

Obj = layer(prev,1,activation = func,isOutput = True)
LayerList.append(Obj)

'''
layer1 = 1024
layer2 = 512
layer3 = 64
layer4 = 1


print('Creating Different Layers ..... ')
Obj1 = layer(layer1,layer2,activation = "sigmoid",keyProb = 1)
Obj2 = layer(layer2,layer3,activation = "sigmoid",keyProb = 1)
Obj3 = layer(layer3,layer4,activation = "sigmoid",isOutput = True)
#Obj3 = layer(layer3,layer4,False)

LayerList = list()
LayerList.append(Obj1)
LayerList.append(Obj2)
LayerList.append(Obj3)
'''
run(TrainSetX,TrainSetY,TestSetX,TestSetY,LayerList,DropOut[0])
#run(TrainSetX.astype(np.float128),TrainSetY.astype(np.float128),TestSetX.astype(np.float128),TestSetY.astype(np.float128),LayerList)
