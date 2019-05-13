'''

Name :- Siddharth Nahar
Entry No :- 2016csb1043
Date :- 20/10/18

Purpose :- 

	1. This contains layer structure for eachLayer
	2. Initialise Weights in linear space in [-0.01 ,0.01]
	3. Bias Terms are zero in short We don't need to consider Weights for bias terms

'''

import numpy as np
from Shift import shift
from Activation import activate

class layer:

	#Constructor for Layer 
	'''
		1. Input Layer Size and Output Layer Size for Theta
		2. Boolean sigmoid if activation is required or not		
	'''

	def __init__(self, inputLayerSize, outputLayerSize, activation = None, keyProb = 1, isOutput = False):

		#Initialising Layer Parameters as layer size
		self.inputLayerSize = inputLayerSize
		self.outputLayerSize = outputLayerSize
		self.activation = activation
		self.keyProb = keyProb
		self.isOutput = isOutput
		#Initialise weights of layer as linear spacing between -0.01 to 0.01
		self.Theta = np.linspace(-0.01,0.01,inputLayerSize*outputLayerSize,dtype = np.float64)


		#self.Theta = np.random.uniform(-0.01,0.01,inputLayerSize*outputLayerSize)
		#Initialise bias weights = 0
		Bias = np.zeros((outputLayerSize,1),dtype = np.float64)

		#Randomly shuffle Theta weights
		#np.random.shuffle(self.Theta)
		self.Theta = self.Theta.reshape((outputLayerSize,inputLayerSize))

		#Add a column of 0 to Theta
		self.Theta = np.append(Bias, self.Theta,axis = 1)
		#print('Layer has Theta Shape as ' + str(self.Theta.shape))

	def __str__(self):

		print("Layer Attributes : ")
		print('Theta Shape ' + str(self.Theta.shape))
		print('Drop Out Percentage : '+str(type(self.keyProb)) + ' '+str(self.keyProb))
		print('Activation Function : '+self.activation)

		return ""

	'''
		1.Performs Forward PAss
		2. a = X@W.T and take activation
		3. u = Dropout as np.random.binomial
	'''

	#Forward Pass for LAyer 
	def forwardPass(self,X):

		#Forward Pass Z = T@X X must be complete matrix
		Z = X@self.Theta.transpose()

		#Multiply By random Binomial for DropOut implementation
		u = np.random.binomial(1,self.keyProb,Z.shape)
		Z = Z*u

		a = None
	
		#If Activation Layer is Present take its activation
		if(self.isOutput == True):
			#If Output has no Activation then just pass a
			a = Z
	
		else:
			#Get Activation layer Output and implement Dropout
			a = activate(Z, self.activation, False)
			a = a*u
			#Append Bias to output
			Bias = np.ones((len(X),1))
			a = np.append(Bias,a,axis = 1)

		
		#print(str(Z.shape) + " " + str(a.shape))
		return Z,a

	'''
		1. Computes Layer gradient
		2. del(3) = del(4)*W
	'''
	#Get Layer Gradient for Further use
	def layerGradient(self,delta_l,Z):

		#Compute Gradient for each layer del = del(l+1)@T

		delta_l1 = delta_l@self.Theta

		#Calculate Gradient of Layer of activation
		grad = activate(Z, self.activation, True)		

		Bias = np.zeros((len(Z),1))
		grad = np.append(Bias,grad,axis = 1)
	
		#Multiply by Gradient of Activation Function
		delta_l1 =  delta_l1*grad

		#Remove bias Term from Delta
		delta_l1 = delta_l1[:,1:]
	
		#Shift for accomodating nan,inf to 0
		delta_l1 = shift(delta_l1)
		#print("Gradient Shape : "+str(delta_l1.shape))
		return delta_l1
