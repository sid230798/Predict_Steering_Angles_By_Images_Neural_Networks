'''


Name :- Siddharth Nahar
Entry No :- 2016csb1043
Date :- 20/10/18

Purpose :- 

	1. Compensate nan and inf values

'''

import numpy as np
from numpy import inf

def shift(X):

	X[X == inf] = 0
	X[X == -inf] = 0
	X[np.isnan(X)] = 0

	return X
