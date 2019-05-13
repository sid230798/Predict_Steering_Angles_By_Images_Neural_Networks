'''

Name :- Siddharth Nahar
Entry No :- 2016csb1043
Date :- 20/10/18

Purpose :- 

	1. PreProcessing function for Data
	2. Normalize each Image
	3. Standardize Train Set get mean and Deviation

'''

import numpy as np
import cv2

#Normalize Function as (x - mean)/std
def Normalize(TrainX, TestX):

	#Calculation of mean and std about axis = 0
	mean = np.mean(TrainX,axis = 0)
	std = np.std(TrainX,axis = 0)

	#X = (X-mean)/std
	TrainSetX = (TrainX - mean)/std
	TestSetX = (TestX - mean)/std

	return TrainSetX, TestSetX
	
#Standardization function as (x - min)/(max - min)
def Standardize(X):

	minX = np.amin(X, axis = 1).reshape((len(X),1))
	maxX = np.amax(X, axis = 1).reshape((len(X),1))

	return (X - minX)/(maxX - minX)

#Preprocessing functiion to read and convert in nupy array
def preProcess(path):

	#PATH_DIR = "l3/steering/"
	PATH_DIR = path
	fileRead = open(PATH_DIR+"data.txt",'r')

	TotalSetX = None
	TotalSetY = None

	#Read Line from File and Load its content
	for index,line in enumerate(fileRead):

		#print("Index : "+str(index))
		line = line.split()
		
		#Extract file path and Sterring angle
		imageFilePath = PATH_DIR + line[0][2:]
		angle = float(line[1])

		X = cv2.imread(imageFilePath,0)
		X = X.flatten()

		tuples = X.shape

		#Rehaping array from (col,) to (col,1)
		X = np.reshape(X,(1,tuples[0]))
		Y = np.reshape([angle],(1,1))

		if(index == 0):
			TotalSetX = X
			TotalSetY = Y
		else:
			TotalSetX = np.concatenate((TotalSetX,X))
			TotalSetY = np.concatenate((TotalSetY,Y))


	#Standardize the Data
	TotalSetX = Standardize(TotalSetX)

	#Create Array ofIndex for Random Sampling
	inx = np.arange(len(TotalSetX))
	np.random.shuffle(inx)

	#Create Training set Sample size = 80%
	TrainSize = int(0.8*len(TotalSetX))

	#Split index according to size
	inx1 = inx[:TrainSize]
	inx2 = inx[TrainSize:]

	#Create Train Set and Test Set
	TrainSetX = TotalSetX[inx1]
	TrainSetY = TotalSetY[inx1]

	TestSetX = TotalSetX[inx2]
	TestSetY = TotalSetY[inx2]

	#Normalize Train and Set x by mean and deviation
	TrainSetX, TestSetX = Normalize(TrainSetX, TestSetX)

	return TrainSetX,TrainSetY,TestSetX,TestSetY
