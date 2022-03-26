import sys

#http://www.scipy.org/
try:
	import numpy as np
	from numpy import dot
	from numpy.linalg import norm
except:
	print("Error: Requires numpy from http://www.scipy.org/. Have you installed scipy?")
	sys.exit() 

def removeDuplicates(list):
	""" remove duplicates from a list """
	return set((item for item in list))


def cosine(vector1, vector2):
	""" related documents j and q are in the concept space by comparing the vectors :
		cosine  = ( V1 * V2 ) / ||V1|| x ||V2|| """
	return float(dot(vector1,vector2) / (norm(vector1) * norm(vector2)))

def Euclidean(vector1, vector2):
    array1 = np.array(vector1)
    array2 = np.array(vector2)
    return np.sum((array1 - array2) ** 2) ** 0.5

