import numpy as np
from numpy import linalg as LA
import cv2
import cv2.aruco as aruco
import time
from scipy.interpolate import griddata
import argparse

with np.load('HD310.npz') as X:
	mtx, dist = [X[i] for i in ('mtx','dist')] 


 
err_tol = 1e-2
tra_tol = 1e-2
ang_tol = 1e-2
max_iter_GN = 2000
approx_points = 100



edge_pts_in_img_sp = [0]*13
aruco_images = [0]*13
aruco_images_int16 = [0]*13
img_pnts = [0]*13
for i in range(1,13):
	edge_pts_in_img_sp[i] = np.loadtxt("thick_edge_coord_R3/id_{}.txt".format(i),delimiter=',',dtype=np.float32)
	aruco_images[i]= cv2.imread("aruco_images_mip_maps/res_38_{}.jpg".format(i),0)
	img_pnts[i] = np.loadtxt("thick_edge_coord_pixels/id_{}.txt".format(i),delimiter=',',dtype='int16')
	aruco_images_int16[i] = np.int16(aruco_images[i])


cv2.imshow("aruco_images[{}]".format(0),aruco_images[0])
cv2.waitKey(0)
aruco_id = 8



frame = cv2.imread("synth_img_{}_transl_200.jpg".format(aruco_id))
frame_gray = cv2.imread("synth_img_{}_transl_200.jpg".format(aruco_id),0)
frame_gray_draw = np.copy(frame_gray)
frame_gray_int16 = np.int16(frame_gray_draw)


# here b is a matrix of coordinates with respect to the centre of the marker  
b = edge_pts_in_img_sp[aruco_id]
b[:,2] = 0.
b[:,3] = 1
n = b.shape[0]
interval = n/approx_points


print n,'n original'
b = b[0::interval,:]
img_pnts[aruco_id] =img_pnts[aruco_id][0::interval,:]
n = b.shape[0]
print n,'n shortened'


b_temp = b[:,0:3].reshape(n,3).astype(np.float32) 


pose_corr = np.array([0.,0.,0.,0.,0.,200.])
pose_pert = pose_corr + np.array([0.1,0.1,0.1,np.random.rand(),np.random.rand(),20.])*1

def DPR_GN(pose_pert,frame_gray):
	t_start = time.time()
	for ii in range(max_iter_GN):
		proj_points, duvec_by_dp_all = cv2.projectPoints( b_temp, pose_pert[0:3], pose_pert[3:6], mtx, dist)

	 	################ we have pixel values of all the points on the aruco marker's .png image.
		################ In this section, I project them all on the pixel plane and select the integer ones only. 
		proj_points_int = np.ndarray.astype(proj_points,int)
		n1 = proj_points_int.shape[0]
		proj_points_int = proj_points_int.reshape(n1,2) ### TODO : reshape takes time
		# proj_points_int = np.unique(proj_points_int,axis=0)
		proj_points = proj_points.reshape(n,2)
		################ EOS 
		 


		### interpolation at pixel values
		######### interpolation of duvec_by_dp_all  Jacobian components
		du_by_dp = griddata(proj_points,duvec_by_dp_all[0::2,0:6],(proj_points_int[:,0],proj_points_int[:,1]), method = 'nearest')
		dv_by_dp = griddata(proj_points,duvec_by_dp_all[1::2,0:6],(proj_points_int[:,0],proj_points_int[:,1]), method = 'nearest')
		n_int = proj_points_int.shape[0]



		dI_by_dv,dI_by_du = np.gradient(frame_gray.astype('int16'))


		dI_by_dp = np.zeros((n_int,6))
		for i in range(n_int):
			ui,vi = proj_points_int[i,0], proj_points_int[i,1]
			dI_by_dp[i,:] = dI_by_du [vi,ui] * du_by_dp[i] + \
				 		dI_by_dv [vi,ui] * dv_by_dp[i]

		 
		temp = proj_points.shape[0]
		proj_points = proj_points.reshape(temp,2)
		 
	 
		for i in range(n_int):
			center = tuple(np.ndarray.astype(proj_points_int[i,:],int))
			cv2.circle( frame_gray_draw, center , 1 , max(127-ii,0), -1)
			center_2 = tuple(np.ndarray.astype(np.array([img_pnts[aruco_id][i,0]+60,img_pnts[aruco_id][i,1]+60]),int))
			cv2.circle( aruco_images[aruco_id], center_2 , 5 , 127, -1)

		# print np.linalg.inv(dI_by_dp.T.dot(dI_by_dp)), "inv(dI_by_dp.T.dot(dI_by_dp))"
		y = aruco_images_int16[aruco_id][img_pnts[aruco_id][:,1]+60,img_pnts[aruco_id][:,0]+60]
		f_p = frame_gray_int16[proj_points_int[:,1],proj_points_int[:,0]]
		err = (y - f_p )/float(n_int)

		dp = np.linalg.inv(dI_by_dp.T.dot(dI_by_dp)).dot(dI_by_dp.T.dot(err))
		# print pose_pert,"p"
		# print dp,"dp"
		# print pose_pert+dp,"p+dp"
		# print pose_corr,"gt"
		# print np.square(err).sum(),"np.square(err)"	
		# print ""
		# print ""
		# for i in range(n):
		# 	center = tuple(np.ndarray.astype(proj_points[i,:],int))
		# 	cv2.rectangle(frame_gray,center,center,127)


		pose_pert = pose_pert + dp
		if (np.square(err).sum() <= err_tol) and ((np.square(dp[0:3]).sum() <= ang_tol) and (np.square(dp[3:6]).sum() <= tra_tol)):
			print ii,"GN iterations"
			print np.square(err).sum() ,"np.square(err).sum()"
			print np.square(dp[0:3]).sum(), "ang_increment"
			print np.square(dp[3:6]).sum(), "tra_increment"
			print "converged"
			break

	t_tot = time.time()-t_start
	print t_tot,"t_GN" 
	print aruco_images[aruco_id].dtype,"aruco_images[aruco_id].dtype"
	cv2.imshow("frame_gray_draw",frame_gray_draw)
	cv2.imshow("frame_gray",frame_gray)
	cv2.imshow("aruco_images[7]",aruco_images[aruco_id])
	cv2.waitKey(0)
	return pose_pert 


print "result" ,DPR_GN(pose_pert,frame_gray)

