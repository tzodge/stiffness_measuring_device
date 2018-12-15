import cv2
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

''' FOR HELP WIITH THE OPTIMISATION CODE FOLLOW THE LINKS=
https://docs.scipy.org/doc/scipy-0.14.0/reference/tutorial/optimize.html#unconstrained-minimization-method-brent'''

# getting the initial estimate

def Rx(th):
	th = -th
	th = np.pi*th/180
	c, s = np.cos(th) , np.sin(th)
	R = np.array(((1,0, 0, 0), (0, c, -s, 0), (0, s, c, 0), (0, 0 ,0, 1) ))
	return R

def Ry(th):
	th = -th
	th = np.radians(th)
	c, s = np.cos(th) , np.sin(th)
	R = np.array(((c,0, s, 0), (0, 1, 0, 0), (-s, 0, c, 0), (0, 0 ,0, 1) ))
	return R

def Rz(th):
	th = -th
	th = np.radians(th)
	c, s = np.cos(th) , np.sin(th)
	R = np.array(((c,-s, 0, 0), (s, c, 0, 0), (0, 0, 1, 0), (0, 0 ,0, 1) ))
	return R

def col(a):
	a = a.reshape(len(a),1)
	return a

def TranslateFrameby(x):
	x = -x
	T = np.array(((1,0, 0, x[0]), (0, 1, 0, x[1]), (0, 0, 1, x[2]), (0, 0 ,0, 1) ))
	return T


#defining the empty list containing transformation from a particular ID to central coordinate system
#refer to the CAD model 
TransfMat = []






theta = 63.436
phi = 72.00
ID3toCentTranslV = np.array([0,19.920305,9.959687,1])
ID3toCentTranslM = TranslateFrameby(ID3toCentTranslV )
ID3toCent = np.matmul(ID3toCentTranslM,Rx(-(180-theta)))


ID6toCentTranslV = np.array([0,-19.920305,-9.959687,1])
ID6toCentTranslM = TranslateFrameby(ID6toCentTranslV)
ID6toCent = np.matmul(ID6toCentTranslM,Rx(theta))
print ""
print "for " , theta
print np.matmul(np.linalg.inv(ID3toCent),ID6toCent)
print "...."




# for i in range (0,100):
# 	theta = 63.435 + 0.0001*i
# 	phi = 72.00
# 	ID3toCentTranslV = np.array([0,19.920305,9.959687,1])
# 	ID3toCentTranslM = TranslateFrameby(ID3toCentTranslV )
# 	ID3toCent = np.matmul(ID3toCentTranslM,Rx(-(180-theta)))


# 	ID6toCentTranslV = np.array([0,-19.920305,-9.959687,1])
# 	ID6toCentTranslM = TranslateFrameby(ID6toCentTranslV)
# 	ID6toCent = np.matmul(ID6toCentTranslM,Rx(theta))
# 	print ""
# 	print "for " , theta
# 	print np.matmul(np.linalg.inv(ID3toCent),ID6toCent)
# 	print "...."



##generates the transformaation of frames 1 to 10 wrt the central
## coordinate system

TransfMat.append(np.identity(4))
 
for i in range (1,6):
	TransfMat.append(np.matmul(Rz(72*(3-i)),ID3toCent))
	# np.matmul(Rz(72*(3-i)),ID3toCent)
	# TransfMat[i][0:3,3] = np.matmul(TransfMat[i][0:3,0:3], np.reshape(TransfMat[i][0:3,3],(3,1))) 

	print i

for i in range (6,11):
	TransfMat.append(np.matmul(Rz(72*(i-6)),ID6toCent))
	# TransfMat[i][0:3,3] = np.matmul(TransfMat[i][0:3,0:3], np.reshape(TransfMat[i][0:3,3],(3,1))) 
	
	print i


#for ID 11  

ID11toCentTranslV = np.array([0,0,-22.27137,1])
ID11toCentTranslM = TranslateFrameby(ID11toCentTranslV )
TransfMat.append(np.matmul(ID11toCentTranslM,Rz(180)))

#for ID 12
ID12toCentTranslV = np.array([0,0,22.27137,1])
ID12toCentTranslM = TranslateFrameby(ID12toCentTranslV )
TransfMat.append((np.linalg.multi_dot((ID12toCentTranslM,Ry(180),Rz(36)))))

print TransfMat[12]

np.save("TransfMatIDtoCent",TransfMat)