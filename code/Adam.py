'''

Name :- Siddharth Nahar
Entry No :- 2016csb1043
Date :- 20/10/18

Purpose :- 

	1. This contains Adam's implementation
	2. It updates learning rate.

	Adam's Optimizer for Learning Rate.
	Alogrithm is implemented below

'''

import numpy as np

#Contains it self parameters to update
class adam:

	#Default Constructor for adam class which takes input as Theta.shape
	def __init__(self,Shape):

		#Default Parameters beta1,beta2,mt,vt and epsilon
		self.t = 0
		self.b1 = 0.9
		self.b2 = 0.999
		self.e  = 0.00000001

		#First Bias Estimate
		self.m  = np.zeros(Shape, dtype = np.float64)

		#Second bias estimate
		self.v  = np.zeros(Shape, dtype = np.float64)

	def optim(self,ThetaGradient):

		
		#Optimizer of Learning Rate
		self.t = self.t + 1

		#Update Equateion of first moment estimate
		self.m = self.b1*self.m + (1 - self.b1)*ThetaGradient

		#Update Equateion of second moment estimate
		self.v = self.b2*self.v + (1 - self.b2)*np.square(ThetaGradient)

		mt = self.m/(1 - self.b1**self.t)
		vt = self.v/(1 - self.b2**self.t)

		vt = np.sqrt(vt)

		#Updated Gradient and returened for Theta Update
		Gradient = mt/(vt + self.e)

		return Gradient
