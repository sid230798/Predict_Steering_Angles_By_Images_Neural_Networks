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
import random
from Shift import shift
import matplotlib.pyplot as plt
from Adam import adam

#Forward Pass Function for Total Layer
def performForwardPass(LayerStructure,X):

	#Z,A list to store layer 2,3 and output layer outputs
	Z = list()
	A = list()

	#Z,A contains all 4 layers of z and a
	Z.append(X)
	A.append(X)

	a = X

	for index,obj in enumerate(LayerStructure):

		#For Each Layer Object perform pass
	
		# z has dimension 32*1 and a has dimension a bias 1 is addded 33*1
		z,a = obj.forwardPass(a)

		Z.append(z)
		A.append(a)

	return Z,A


def updateWeights(X,Y,LayerList,alpha,DropOut,isOptimized,AdamList):

	#print('Training Started ...... ')
	n = len(LayerList)

	Bias = np.ones((len(X),1))

	u1 = np.random.binomial(1,DropOut,size = X.shape)

	x = X*u1
	x = np.append(Bias,X,axis = 1)

	#print(x.shape)
	#Gradient of three layers and initializing with zeros of Theta shape
	ThetaGradient = list()

	#Initialising ThetaGradient with Theta.shape
	for index,Obj in enumerate(LayerList):

		dt = np.zeros(Obj.Theta.shape,dtype = np.float64)
		#print('Initialising Gradient of Theta as ' + str(dt.shape))
		ThetaGradient.append(dt)

	#For each training example update cummulative gradient

	Z,A = performForwardPass(LayerList,x)

	Err = A[n] - Y

	for i in range(n-1,-1,-1):

		Obj = LayerList[i]

		ThetaGradient[i] = Err.transpose()@A[i]
		ThetaGradient[i] = shift(ThetaGradient[i])
		#ThetaGradient[i] = shift(ThetaGradient[i])
		if(i > 0):
			Err = Obj.layerGradient(Err,Z[i])		

	
	for index,Obj in enumerate(LayerList):

		if(isOptimized == True):
			Obj.Theta = Obj.Theta - alpha*AdamList[index].optim(ThetaGradient[index]/len(X))
		else:
			Obj.Theta = Obj.Theta - alpha*ThetaGradient[index]/len(X)

#Loss function as Square Loss
def computeCost(TrainSetX,TrainSetY,LayerList):

	#Add a bias 1 for first layer
	Bias = np.ones((len(TrainSetX),1))

	x = np.append(Bias,TrainSetX,axis = 1)

	#Performs Forward pass and Takes output as z,a of layers
	Z,A = performForwardPass(LayerList,x)

	#Err = 1/2*(o-y)^2
	n = len(LayerList)
	Err = A[n] - TrainSetY

	Err = shift(Err)

	Err = Err.astype(dtype = np.float128)
	Err = np.square(Err)/(2*len(TrainSetX))

	#Sum of all Errords
	Err = np.sum(Err)

	return Err

#Shuffle X,Y for every epoch
def shuffle(X,Y):

	inx = np.arange(len(X))
	np.random.shuffle(inx)

	TrainSetX = X[inx]
	TrainSetY = Y[inx]

	return TrainSetX,TrainSetY

def Plot(X,Y,Itr,Title,maxIter,Fig):

	plt.plot(Itr,X,'b',label = 'Training Error')
	plt.plot(Itr,Y,'r',label = 'Validation Error')
	plt.title(Title)
	plt.xlabel("Epochs")
	plt.ylabel("Error")
	plt.gca().set_xlim([0,maxIter])
	plt.gca().set_ylim([0.0,0.20])
	plt.legend(loc = 'upper right')
	plt.savefig(Fig)
	plt.show()

def getListOfAdam(LayerList):

	AdamList = list()
	for Obj in LayerList:

		AdamObj = adam(Obj.Theta.shape)
		AdamList.append(AdamObj)

	return AdamList

def run(X,Y,TestSetX,TestSetY,LayerList,DropOut):

	'''
	miniBatchSize = 128
	DropOut = 0
	alpha = 0.01
	maxIter = 1000
	'''
	miniBatchSize = int(input('Enter the Batch Size for Training : '))
	alpha = float(input('Enter the Learning Rate : '))
	maxIter = int(input('Enter number of Epochs for training : '))

	TotExamples = len(X)
	N_Batches = int(TotExamples/miniBatchSize)

	TotalPoints = min(200,maxIter)
	Interval = int(maxIter/TotalPoints)

	TrainError = list()
	TestError = list()
	Iterations = list()

	AdamList = getListOfAdam(LayerList)
	
	val = computeCost(X,Y,LayerList)
	val1 = computeCost(TestSetX,TestSetY,LayerList)
	print("Cost after Iteration 0")
	print("Training Error : " +str(val))
	print("Validation Error : " +str(val1))

	Iterations.append(0)
	TrainError.append(val)
	TestError.append(val1)
	print('Started Training with Batch Size and Learning Rate ' + str(miniBatchSize) + ' '+str(alpha))
	for i in range(0,maxIter):

		#print('Iteration ' + str(i+1))
		
		TrainSetX,TrainSetY = shuffle(X,Y)
		
		for j in range(0,N_Batches+1):
			#print("Batch j Started " + str(j))

			
			startIndex = j*miniBatchSize
			endIndex = min(j*miniBatchSize + miniBatchSize,TotExamples)

			if(startIndex > TotExamples-2):
				continue

			SampleX = TrainSetX[startIndex:endIndex]
			SampleY = TrainSetY[startIndex:endIndex]

			updateWeights(SampleX,SampleY,LayerList,alpha,DropOut,True,AdamList)
			#print("Batch Ended")

		val = computeCost(X,Y,LayerList)

		val1 = computeCost(TestSetX,TestSetY,LayerList)

		print("Cost after Iteration "+str(i+1))
		print("Training Error : " +str(val))
		print("Validation Error : " +str(val1))
		if((i+1)%Interval == 0):
			Iterations.append(i+1)
			TrainError.append(val)
			TestError.append(val1)
	
	Qus = "Adam.jpg"
	Title = "Training vs Validation Error" + "\nLearning Rate : "+str(alpha)+"\nMiniBatchSize : "+str(miniBatchSize) +"\nKeyProb : "+str(DropOut)
	Plot(TrainError,TestError,Iterations,Title,maxIter,Qus)
