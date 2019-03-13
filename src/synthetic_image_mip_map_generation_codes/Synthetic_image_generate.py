#Used this code to confirm that the tvec and rvec given by the 
 # estimatePoseSingleMarkers is of the marker frame wrt the camera frame


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

	
def RodriguesToTransf(x):
	#input (6,) (rvec,tvec)
	x = np.array(x)
	rot,_ = cv2.Rodrigues(x[0:3])
	trans =  np.reshape(x[3:6],(3,1))
	Trransf = np.concatenate((rot,trans),axis = 1)
	Trransf = np.concatenate((Trransf,np.array([[0,0,0,1]])),axis = 0)
	return Trransf
	
	

def draw_synth_img(frame,pose,id_no):

	frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	frame_draw_white = np.copy(frame_gray)
	frame_draw_white[:,:] = 255 

	marker_id_str = "aruco_images_mip_maps/res_"+ res + "_{}.jpg".format(id_no)  
	aruco_img = cv2.imread(marker_id_str)
	aruco_img_gray = cv2.cvtColor(aruco_img,cv2.COLOR_BGR2GRAY)
	

	n = aruco_img.shape[0]
	Intens_val = aruco_img_gray.reshape(n**2,1)

	scale_fact = 1.2*marker_size_in_mm/n

	img_points_3d = np.zeros((4,n**2))
	img_points_3d[3,:] = 1
	print img_points_3d.shape,"img_points_3d.shape"	
	######## this for loop can be made more efficient. all that it does is [x = u-n/2] and [y = m/2-v]
	for i in range(n):
		img_points_3d[1,n*i:n*(i+1)] = (n/2.0 - i) *scale_fact
		img_points_3d[0,n*i:n*(i+1)] = (np.arange(0,n)-n/2.0)*scale_fact

	print img_points_3d[:,0:9]	
	tf_mat = RodriguesToTransf(pose.reshape(6,))	
	transformed_points = tf_mat.dot(img_points_3d)


	proj_points,_ = cv2.projectPoints(transformed_points[0:3,:].T,np.zeros((1,3)),np.zeros((1,3)),mtx,dist)
	proj_points_int = np.ndarray.astype(proj_points,int)

	n1 = proj_points_int.shape[0]
	proj_points_int = proj_points_int.reshape(n1,2)
	proj_points_int = np.unique(proj_points_int,axis=0)
	
	proj_points = proj_points.reshape(n*n,2)
	### interpolation at pixel values
	Intens_val_interpolated = griddata(proj_points,Intens_val,(proj_points_int[:,0],proj_points_int[:,1]), method = 'cubic')

	for i in range(proj_points_int.shape[0]):
		cv2.circle( frame_draw_white, tuple(np.ndarray.astype(proj_points_int[i,:],int)) , 1 , Intens_val_interpolated[i], -1)
		cv2.circle( frame_gray, tuple(np.ndarray.astype(proj_points_int[i,:],int)) , 1 , Intens_val_interpolated[i], -1)
	

	######### following lines confirm that the camera matrix is not accurate.  
	# frame_size = frame_gray.shape
	# frame_gray[frame_size[0]/2-5:frame_size[0]/2+5,frame_size[1]/2-5:frame_size[1]/2+5] = 127
	# frame_draw_white[frame_size[0]/2-5:frame_size[0]/2+5,frame_size[1]/2-5:frame_size[1]/2+5] = 127



	cv2.imshow("frame_draw_white",frame_draw_white)
	cv2.imshow("frame_gray",frame_gray)
	cv2.waitKey(0)




sub_pix_refinement_switch =1
marker_size_in_mm = 17.78

with np.load('PTGREY.npz') as X:
	mtx, dist = [X[i] for i in ('mtx','dist')] 


frame = cv2.imread("sample_image_pntgrey.png")

pose = np.array([-np.pi,0.,0., 0.,0.,200.])
id_no = 7
print mtx
# mtx = np.array([[813,0, 320],[0, 813, 240],[0, 0, 1] ], dtype=np.float)
res  = "600"
 

# draw_synth_img(frame,pose,id_no)
draw_synth_img(frame,pose,7)
# draw_synth_img(frame,pose,9)

