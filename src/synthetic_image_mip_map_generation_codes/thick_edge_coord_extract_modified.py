
import numpy as np
from numpy import linalg as LA
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
from mpl_toolkits.mplot3d import Axes3D
import transforms3d as tf3d
import time
from scipy.optimize import minimize, leastsq
from scipy import linalg
from scipy.interpolate import griddata


def marker_edges (marker_id, dwn_smpl_to,thickness):
	
	marker_id_str = "aruco_images" + "/{}.jpg".format(marker_id)  
	gray = cv2.imread(marker_id_str)
	a = cv2.Canny(gray,100,200,apertureSize = 5)
	a[0,:] = 255
	a[-1,:] = 255
	a[:,0] = 255
	a[:,-1] = 255
	size_img = gray.shape[0]
	a_before = np.copy(a)
	a = np.flipud(a).T #do not change this!!! #            
	scale = a.shape[0]/marker_size_in_mm
	a_shift = a.shape[1]/(2*scale)
	b = np.asarray(np.nonzero(a))
	z = np.zeros((1,b.shape[1]))
	tr_dash = np.ones((1,b.shape[1]))

	b_before = np.asarray(np.nonzero(a))
	######### to find intensity in the orginal image
	# b_before = np.fliplr(b)
	temp = np.vstack((tr_dash,z))*size_img
	b_before = abs(b_before-temp)
	######### to find intensity in the orginal image

	b = np.vstack ((b,z,tr_dash))
	print b.shape,"b.shape"
	b_thick_temp = np.zeros((4, (2*thickness+1)*b.shape[1]))
	b_thick_temp[3,:] = 1 

#	b_before_thick_temp = np.zeros((4, (2*thickness+1)*b_before.shape[1]))
	#b_before_thick_temp[3,:] = 1

	n = b.shape[1]
	for i in range(-thickness, thickness+1):
		for j in range(-thickness, thickness+1):
			
			b_thick_temp[0,(i+thickness)*n:(i+thickness+1)*n ] = b[0,:] + i				
			b_thick_temp[1,(j+thickness)*n:(j+thickness+1)*n ] = b[1,:] + j	

			#b_before_thick_temp[0,(i+thickness)*n:(i+thickness+1)*n ] = b_before[0,:] + i				
			#b_before_thick_temp[1,(j+thickness)*n:(j+thickness+1)*n ] = b_before[1,:] + j			

	interval =  int(b_thick_temp.shape[1]/dwn_smpl_to)
	b_thick_temp = b_thick_temp[:,0::interval]
	Rx_pi = np.eye(4)
	Rx_pi[1,1],Rx_pi[2,2] = -1,-1
	#b_before_thick_temp = #b_before_thick_temp[:,0::interval]
	temp_arr = np.array([[0],[0],[0],[0]])
	b_thick_temp_as_in_img = Rx_pi.dot(b_thick_temp-a_shift*scale) +a_shift*scale
	print b_thick_temp.shape,"b_thick_temp.shape"
	return b_thick_temp/scale - a_shift, b_thick_temp_as_in_img #b_before_thick_temp








###################################################################
#### can be used for debugging or to check what's happening #######
#### marker_edges (marker_id, dwn_smpl_to,thickness):##############
################################################################### 

with np.load('HD310.npz') as X:
	mtx, dist = [X[i] for i in ('mtx','dist')] 

marker_size_in_mm = 17.78

for i in range(1,13):
	thick_edge_coord_R3,b_thick_temp = marker_edges (i, 10000, 3)
	np.savetxt("thick_edge_coord_R3/id_{}.txt".format(i),thick_edge_coord_R3.T , delimiter=',')
	np.savetxt("thick_edge_coord_pixels/id_{}.txt".format(i),b_thick_temp.T,fmt='%d' , delimiter=',')
# b = marker_edges(8,10000,3)
# b[2,:] = 200.0             ### transforming all the points in z direction by 200 mm 

# print b[0:3,0:5].T, "b[0:3,:].T"
# # print b?
# print b[0,0].dtype,"b.dtype"


# frame = cv2.imread("synth_img_8_transl_200.jpg")

# proj_points,_ = cv2.projectPoints(b[0:3,:].T,np.zeros((1,3)),np.zeros((1,3)),mtx,dist)
# temp = proj_points.shape[0]
# proj_points = proj_points.reshape(temp,2)
# pix_integer = np.ndarray.astype(proj_points,int)
# # print pix_integer.shape, " pix_integer.shape"
# # print pix_integer,"pix_integer"
# intensities = frame[pix_integer[:,1],pix_integer[:,0]]

# print intensities

# for i in range(b.shape[1]):
# 	center = tuple(np.ndarray.astype(proj_points[i,:],int))
# 	cv2.circle( frame, center , 1 , (0,255,0), -1)
# 	cv2.rectangle(frame,center,center,(0,0,255))

# # ############################################

# # b = marker_edges(8,1000,0)
# # b[2,:] = 200.0
# # print b[0:3,:].T.shape, 'b[0:3,:].T.shape'

# # proj_points,_ = cv2.projectPoints(b[0:3,:].T,np.zeros((1,3)),np.zeros((1,3)),mtx,dist)
# # temp = proj_points.shape[0]
# # proj_points = proj_points.reshape(temp,2)


# # for i in range(b.shape[1]):
# # 	center = tuple(np.ndarray.astype(proj_points[i,:],int))
# # 	# cv2.circle( frame, center , 1 , (0,0,255), -1)
# # 	cv2.rectangle(frame,center,center,(255,0,0))

# # ################

# cv2.imshow("frame",frame)
# cv2.waitKey(0)
