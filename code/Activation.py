'''

Name :- Siddharth Nahar
Entry No :- 2016csb1043
Date :- 20/10/18

Purpose :- 

	1. Stable Sigmoid function for implementation in Hidden Layer
	2. Tanh function for hidden layer
	3. Re-LU for activation layer

'''

#Importing Libraires for Activation Functions of Different layers
from scipy.special import expit
import numpy as np

#Sigmoid Activation layer
def sigmoid(z, derivation):

	#Sigmoid = 1/(1 + exp(-z))
	a = expit(z)

	#Derivation = a*(1-a)
	if(derivation == True):
		return a*(1 - a)

	return a

#TanH activation Layer
def tanH(z, derivation):

	#TanH = tanH(z)
	a = np.tanh(z)

	#Derivation = 1-tanh(z)**2
	if(derivation == True):
		return 1 - np.square(a)

	return a

#ReLU activation Layer
def ReLu(z, derivation):

	#Derivation if(x>0) = 1 else 0
	if(derivation == True):
		return 1.0*(z > 0)

	#Relu if(x>0) = x else 0
	return z*(z > 0)

#Select Activation layer
def activate(z, activationLayer, derivation):

	#Select the activation layer by name and return its value
	if(activationLayer.lower() == "sigmoid"):
		return sigmoid(z, derivation)
	elif(activationLayer.lower() == "tanh"):
		return tanH(z, derivation)
	else:
		return ReLu(z, derivation)
