import cv2
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
from  scipy.optimize import minimize_scalar


from scipy.optimize import minimize

''' FOR HELP WIITH THE OPTIMISATION CODE FOLLOW THE LINKS=
https://docs.scipy.org/doc/scipy-0.14.0/reference/tutorial/optimize.html#unconstrained-minimization-method-brent'''

# def rosen(x):
#     """The Rosenbrock function"""
#     return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)



# x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
# res = minimize(rosen, x0, method='nelder-mead',
#                options={'xtol': 1e-8, 'disp': True})

def dist(p1,p2):
	return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 


points = [np.array([1,1]), np.array([1,-1]), np.array([-1,-1]), np.array([-1,1])]


def closest(x):
	total = 0
	for i in range (0,len(points)):
		total = total + dist (x, points[i]) 
		# print type(points[i])

	return total



x0 = np.array([0.2,-0.1]) 

# res = closest (x0)
# print res
res = minimize(closest, x0, method='nelder-mead',
               options={'xtol': 1e-8, 'disp': True})

print(res.x)

# print type(points[2])
