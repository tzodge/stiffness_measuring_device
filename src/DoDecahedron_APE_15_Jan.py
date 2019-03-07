#Used this code to confirm that the tvec and rvec given by the 
 # estimatePoseSingleMarkers is of the marker frame wrt the camera frame


# from __future__ import division
import numpy as np
from numpy import linalg as LA
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
from mpl_toolkits.mplot3d import Axes3D
import transforms3d as tf3d
import time
#from helper import *
from scipy.optimize import minimize, leastsq,least_squares
from scipy import linalg
from scipy.spatial import distance
import rospy
from roscam import RosCam

_R_cent_face = np.load('Center_face_rotations.npy')
_T_cent_face = np.load('Center_face_translations.npy')
frame_gray = np.zeros((100,100, 3), np.uint8)
frame_gray_draw = np.copy(frame_gray)


def draw_3d_point(frame, pose, coord_3d, mtx, dist, col=(255,0,255), rad = 10):
	coord_3d =coord_3d.reshape(1,3)
	rot = pose[0:3].reshape(3,1)
	tra = pose[3:6].reshape(3,1)
	projected_in_pix_sp,_ = cv2.projectPoints(coord_3d,rot,tra,mtx,dist) 
	temp1 = int(projected_in_pix_sp[0,0,0])
	temp2 = int(projected_in_pix_sp[0,0,1])	
	cv2.circle(frame,(temp1,temp2), rad , col, 2)

def slerp(v0, v1, t_array):
    # >>> slerp([1,0,0,0],[0,0,0,1],np.arange(0,1,0.001))
    t_array = np.array(t_array)
    v0 = np.array(v0)
    v1 = np.array(v1)
    dot = np.sum(v0*v1)
    if (dot < 0.0):
        v1 = -v1
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if (dot > DOT_THRESHOLD):
        result = v0[np.newaxis,:] + t_array[:,np.newaxis]*(v1 - v0)[np.newaxis,:]
        result = result/np.linalg.norm(result)
        return result

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0*t_array
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0[:,np.newaxis] * v0[np.newaxis,:]) + (s1[:,np.newaxis] * v1[np.newaxis,:])

####################
def find_tfmat_avg(T_cent_accepted):
	Tf_cam_ball = np.eye(4)

	#### using slerp interpolation for averaging rotations
	sum_tra = np.zeros(3,)
	quat_av = tf3d.quaternions.mat2quat(T_cent_accepted[0][0:3,0:3])
	for itr in range(T_cent_accepted.shape[0]-1):
		quat2 = tf3d.quaternions.mat2quat(T_cent_accepted[itr+1][0:3,0:3])
		quat_av = slerp(quat2,quat_av,[0.5])
		quat_av =quat_av.reshape(4,)

	Tf_cam_ball[0:3,0:3]=tf3d.quaternions.quat2mat(quat_av)


	for itr in range(T_cent_accepted.shape[0]):
		sum_tra = sum_tra + T_cent_accepted[itr][0:3,3]

	sum_tra = sum_tra/T_cent_accepted.shape[0]
	Tf_cam_ball[0:3,3] = sum_tra
	
	return 	Tf_cam_ball
#######################################################

def RodriguesToTransf(x):
	'''
	Function to get a SE(3) transformation matrix from 6 Rodrigues parameters. NEEDS CV2.RODRIGUES()
	input: X -> (6,) (rvec,tvec)
	Output: Transf -> SE(3) rotation matrix
	'''
	x = np.array(x)
	rot,_ = cv2.Rodrigues(x[0:3])
	trans =  np.reshape(x[3:6],(3,1))
	Transf = np.concatenate((rot,trans),axis = 1)
	Transf = np.concatenate((Transf,np.array([[0,0,0,1]])),axis = 0)
	return Transf

def LM_APE_Dodecapen(X,stacked_corners_px_sp, ids, marker_size_in_mm, mtx, dist, flag=False):
	'''
	Function to get the objective function for APE step of the algorithm
	TODO: Have to put it in its respective class as a method (kind attn: Howard)
	Inputs: 
	X: (6,) array of pose parameters [rod_1, rod_2,rod_3,x,y,z]
	stacked_corners_px_sp = Output from Aruco marker detection. ALL the corners of the markers seen stacked in order 
	ids: int array of ids seen -- ids of faces seen

	Output: V = [4*M x 1] numpy array of difference between pixel distances
	'''
	# print(ids)
	corners_in_cart_sp = np.zeros((ids.shape[0],4,3))
	Tf_cam_ball = RodriguesToTransf(X)
	for ii in range(ids.shape[0]):
		Tf_cent_face,Tf_face_cent = tf_mat_dodeca_pen(int(ids[ii]))
		corners_in_cart_sp[ii,:,:] = Tf_cam_ball.dot(corners_3d(Tf_cent_face,marker_size_in_mm)).T[:,0:3]
	
	corners_in_cart_sp = corners_in_cart_sp.reshape(ids.shape[0]*4,3)
	projected_in_pix_sp,_ = cv2.projectPoints(corners_in_cart_sp,np.zeros((3,1)),np.zeros((3,1)),mtx,dist) 
	projected_in_pix_sp = projected_in_pix_sp.reshape(projected_in_pix_sp.shape[0],2)
	n,_=np.shape(stacked_corners_px_sp)
	V = LA.norm(stacked_corners_px_sp-projected_in_pix_sp, axis=1)
	
	if flag is False:
		return V

def tf_mat_dodeca_pen(face_id):
	'''
	Function that looks at the dodecahedron geometry to get the rotation matrix and translation
	TODO: when in class have to pass the dodecahedron geometry to this as a variable
	Inputs: face_id: the face for which the transformation matrices is quer=ries (int)
	Outputs: T_mat_cent_face = transformation matrix from center of the dodecahedron to a face
	T_mat_face_cent = transformation matrix from face (with given face id) to the dodecahedron center
	
	'''
	T_cent_face_curr = _T_cent_face[face_id-1,:,:]
	_R_cent_face_curr = _R_cent_face[face_id-1,:,:]
	T_mat_cent_face = np.vstack((np.hstack((_R_cent_face_curr,T_cent_face_curr)),np.array([0,0,0,1])))
	T_mat_face_cent = np.vstack((np.hstack((_R_cent_face_curr.T,-_R_cent_face_curr.T.dot(T_cent_face_curr))),np.array([0,0,0,1])))
	return T_mat_cent_face,T_mat_face_cent

def corners_3d(tf_mat,m_s):
	'''
	Function to give coordinates of the marker corners and transform them using a given transformation matrix
	Inputs:
	tf_mat = transformation matrix between frames
	m_s = marker size-- edge lenght in mm
	Outputs:
	corn_pgn_f = corners in camara frame
	'''
	corn_1 = np.array([-m_s/2.0,  m_s/2.0, 0, 1])
	corn_2 = np.array([ m_s/2.0,  m_s/2.0, 0, 1])
	corn_3 = np.array([ m_s/2.0, -m_s/2.0, 0, 1])
	corn_4 = np.array([-m_s/2.0, -m_s/2.0, 0, 1])
	corn_mf = np.vstack((corn_1,corn_2,corn_3,corn_4))
	corn_pgn_f = tf_mat.dot(corn_mf.T)
	return corn_pgn_f

def remove_bad_aruco_centers(center_transforms, mtx, dist):

	"""
	takes in the tranforms for the aruco centers

	returns the transforms, centers coordinates, and indices for centers which

	aren't too far from the others

	Input: center_transforms  = N transformation matrices stacked as [N,4,4] numpy arrays
	Output = center_transforms[good_indices, :, :] -> accepted center transforms,
	centers_R3[good_indices, :] = center estimates from accepted center transforms,
	good_indices = accepted ids
	"""
	max_dist = 25 # pixels 

	centers_R3 = center_transforms[:, 0:3, 3]
	projected_in_pix_sp,_ = cv2.projectPoints(centers_R3,np.zeros((3,1)),np.zeros((3,1)), mtx, dist) 
	
	projected_in_pix_sp = projected_in_pix_sp.reshape(projected_in_pix_sp.shape[0],2)


	distances = distance.cdist(centers_R3, centers_R3)
	distances_2 = distance.cdist(projected_in_pix_sp, projected_in_pix_sp)
	# print distances_2,"distances_2"
	# print distances,"distances"
	good_pairs = (distances_2 > 0) * (distances_2 < max_dist)

	good_indices = np.where(np.sum(good_pairs, axis=0) > 0)[0].flatten()

	if good_indices.shape[0] == 0 :
		print('good_indices is none, resetting')
		good_indices = np.array([0, 1]) 

	return center_transforms[good_indices, :, :], centers_R3[good_indices, :], good_indices

########### Alternate approach added by tejas
# def remove_bad_aruco_centers(center_transforms):

# 	"""
# 	takes in the tranforms for the aruco centers

# 	returns the transforms, centers coordinates, and indices for centers which

# 	aren not very skewed wrt the camera 

# 	Input: center_transforms  = N transformation matrices stacked as [N,4,4] numpy arrays
# 	Output = center_transforms[good_indices, :, :] -> accepted center transforms,
# 	centers_R3[good_indices, :] = center estimates from accepted center transforms,
# 	good_indices = accepted ids
# 	"""

# 	centers_R3 = center_transforms[:, 0:3, 3]

# 	# distances = distance.cdist(centers_R3, centers_R3)

# 	# good_pairs = (distances > 0) * (distances < max_dist)

# 	# good_indices = np.where(np.sum(good_pairs, axis=0) > 0)[0].flatten()

# 	# if good_indices.shape[0] == 0 :
# 	# 	print('good_indices is none, resetting')
# 	# 	good_indices = np.array([0, 1]) 


# 	return center_transforms[good_indices, :, :], centers_R3[good_indices, :], good_indices
# ########### Alternate approach added by tejas



# def local_frame_grads (frame_gray, corners, ids): #### by arkadeep 
# 	''' Takes in the frame, the corners of the markers the camera sees and ids of the markers seen. 
# 	Returns the frame gradients
# 	Input: frame_gray --> grayscale frame
# 	corners: stacked as corners[num_markers,:,:] 
# 	ids: ids seen
# 	Output
# 	frame_grad_u and frame_grad_v: matrices of sizes as frame gray. the areas near the markers will 
# 	have gradients of the frame_gray frame in the same locations and rest is 0
# 	'''
# 	dilate_fac =1.2
# 	frame_grad_u = np.zeros((frame_gray.shape[0],frame_gray.shape[1]))
# 	frame_grad_v = np.zeros((frame_gray.shape[0],frame_gray.shape[1]))

# 	for i in range(len(ids)):
# 		expanded_corners = get_marker_borders(corners[i,:,:],dilate_fac)
# 		v_low = int(np.min(expanded_corners[:,1]))
# 		v_high = int(np.max(expanded_corners[:,1])) 
# 		u_low = int(np.min(expanded_corners[:,0]))
# 		u_high = int(np.max(expanded_corners[:,0]))

# 		frame_local = np.copy(frame_gray[v_low:v_high,u_low:u_high]) # not sure if v and u are correct order

# 		A,B = np.gradient(frame_local.astype('float32'))
# 		frame_grad_v[v_low:v_high,u_low:u_high] = np.copy(A) 
# 		frame_grad_u[v_low:v_high,u_low:u_high] = np.copy(B)
# 	return  frame_grad_u, frame_grad_v

def local_frame_grads(frame_gray, cent_R3, pix_width,mtx,dist):
	''' takes in frame, gives gradient of the square centered at dodecahedron center
		with size pix_width x pix_width
	'''
	cent_R3 = cent_R3.reshape(1,3)
	frame_copy_a = np.zeros(frame_gray.shape,dtype=np.float32)
	frame_copy_b = np.zeros(frame_gray.shape,dtype=np.float32)
	cent_pix, _ = cv2.projectPoints( cent_R3, np.zeros((3,1)), np.zeros((3,1)), mtx, dist)
	proj_point_int = np.ndarray.astype(cent_pix,int)	
	proj_point_int = proj_point_int.reshape(2,)
	local_frame = frame_gray[proj_point_int[1]-pix_width/2 : proj_point_int[1]+pix_width/2, 
   							 proj_point_int[0]-pix_width/2 : proj_point_int[0]+pix_width/2]
 	a,b = np.gradient(local_frame)
 
 	frame_copy_a[proj_point_int[1]-pix_width/2 : proj_point_int[1]+pix_width/2, 
   				proj_point_int[0]-pix_width/2 : proj_point_int[0]+pix_width/2]  = a

	frame_copy_b[proj_point_int[1]-pix_width/2 : proj_point_int[1]+pix_width/2, 
   				proj_point_int[0]-pix_width/2 : proj_point_int[0]+pix_width/2]  = b
 
   	frame_grad_show = np.copy(frame_copy_b)
   	cv2.imshow("frame_grad_show",frame_grad_show)
   	cv2.waitKey(1)
   	return frame_copy_a,frame_copy_b			

def marker_edges(ids, downsample, edge_pts_in_img_sp, aruco_images_int16, img_pnts):
	''' Function to give the edge points in the image and their intensities.
	to be called only once in the entire program to gather reference data for DPR
	output: 
	b_edge = [:,:] of size (ids x downsample,2) points on the marker in R3 where the intensities change form [x,y,0] stacked as 
	to be directly used in cv2.projectpoints
	edge_intensities = [:] ordered expected intensity points for the edge points to be used on obj fun of DPR size (ids x downsample)
	'''
	b_edge = []
	edge_intensities_expected = []

	for aruco_id in ids:
		b = edge_pts_in_img_sp[aruco_id[0]]
		n = b.shape[0]
		b[:,2] = 0.
		b[:,3] = 1
		b_shaped = b[0::downsample,0:4].astype(np.float32) 
		b_edge.append(b_shaped)
		img_pnts_curr =img_pnts[aruco_id[0]][0::downsample,:]
		edge_intensities = aruco_images_int16[aruco_id[0]][img_pnts_curr [:,1]+60,img_pnts_curr [:,0]+60]# TODO can we have it in terms of dil_fac
		edge_intensities_expected.append(edge_intensities)
	# ---------------------------------------------------------------

 	########## this part varifies that correct points are sampled from the edges of the aruco images. 
	# for j in range(len(ids)):
	# 	img_pnts_curr =img_pnts[ids[j][0]][0::downsample,:]
	# 	n_int = img_pnts_curr.shape[0]
	# 	print img_pnts_curr.shape,"img_pnts_curr.shape"
	# 	print img_pnts[ids[j][0]].shape,"img_pnts.shape"
	# 	for i in range(n_int):
	# 		center_2 = tuple(np.ndarray.astype(np.array([img_pnts_curr[i,0]+60,img_pnts_curr[i,1]+60]),int))
	# 		cv2.circle( aruco_images[ids[j][0]], center_2 , 3 , 127, -1)
	# 	cv2.imshow("aruco_images[ids_{}]".format(ids[j][0]),aruco_images[ids[j][0]])
	# 	cv2.waitKey(0)	

	# print edge_intensities_expected[0]," edge_intensities_expected[{}]".format(ids[0][0])
	# print ""	
	# print ids[0]
	# print b_edge[0].shape,"b_edge[0].shape"
	# print edge_intensities_expected[0].shape,"edge_intensities_expected[0].shape"
	# print ""	

		# y = aruco_images_int16[aruco_id][img_pnts[aruco_id][:,1]+60,img_pnts[aruco_id][:,0]+60]

	return np.asarray(b_edge) , np.asarray(edge_intensities_expected)

	pass

def get_marker_borders (corners,dilate_fac):
	''' Dilates a given marker from the corner pxl locations in an image by the dilate factor. 
	Returns: stack of expanded corners in the pixel space'''

	cent = np.array([np.mean(corners[:,0]), np.mean(corners[:,1])])
	
	vert_1 = (corners[0,:] - cent)* dilate_fac
	vert_2 = (corners[1,:] - cent)* dilate_fac
	vert_3 = (corners[2,:] - cent)* dilate_fac
	vert_4 = (corners[3,:] - cent)* dilate_fac
	
	expanded_corners = np.vstack((vert_1+cent,vert_2+cent,vert_3+cent,vert_4+cent))
	 
	return expanded_corners

def LM_DPR(X, frame_gray, ids, corners, b_edge, edge_intensities_expected_all, aruco_images, mtx, dist):
	''' Objective function for the DPR step. Takes in pose as the first arg [mandatory!!] and,
	returns the value of the obj fun and jacobial of the obj fun'''    
	Tf_cam_ball = RodriguesToTransf(X)
	borders_in_cart_sp = []
	edge_intensities_expected = []
	for ii in range(ids.shape[0]):
		Tf_cent_face,Tf_face_cent = tf_mat_dodeca_pen(int(ids[ii]))
		borders_in_cart_sp.append((Tf_cam_ball.dot(Tf_cent_face).dot(b_edge[ii].T)).T)
		edge_intensities_expected.append(edge_intensities_expected_all[ii].reshape(edge_intensities_expected_all[ii].shape[0],1))

	stacked_borders_in_cart_sp = np.vstack(borders_in_cart_sp)
	edge_intensities_expected_stacked = np.vstack(edge_intensities_expected)

	proj_points, _ = cv2.projectPoints( stacked_borders_in_cart_sp [:,0:3], np.zeros((3,1)), np.zeros((3,1)), mtx, dist)
	proj_points_int = np.ndarray.astype(proj_points,int)

	proj_points_int = proj_points_int.reshape(proj_points_int.shape[0],2)
	n_int = proj_points_int.shape[0]
	temp = proj_points.shape[0]
	proj_points = proj_points.reshape(temp,2)
	
	# testing_points = 10 
 	# LM_DPR_DRAW(X, frame_gray_draw, ids, corners, b_edge, edge_intensities_expected_all,aruco_images, mtx, dist, 127)

	f_p = frame_gray[proj_points_int[:,1],proj_points_int[:,0]]  # TODO i dont think framegray int16 is needed ? Also 0,1 order changed
	err = (edge_intensities_expected_stacked/1.0 - f_p.reshape(f_p.shape[0],1)/1.0) # this is the error in the intensities
	return  err.reshape(err.shape[0],)

def LM_DPR_DRAW(X, frame_gray, ids, corners, b_edge, edge_intensities_expected_all, aruco_images, mtx, dist, col_gr = 127, rad=1):
	''' Objective function for the DPR step. Takes in pose as the first arg [mandatory!!] and,
	returns the value of the obj fun and jacobial of the obj fun'''    
	Tf_cam_ball = RodriguesToTransf(X)
	borders_in_cart_sp = []
	edge_intensities_expected = []
	for ii in range(ids.shape[0]):
		Tf_cent_face,Tf_face_cent = tf_mat_dodeca_pen(int(ids[ii]))
		borders_in_cart_sp.append((Tf_cam_ball.dot(Tf_cent_face).dot(b_edge[ii].T)).T)
		edge_intensities_expected.append(edge_intensities_expected_all[ii].reshape(edge_intensities_expected_all[ii].shape[0],1))

	stacked_borders_in_cart_sp = np.vstack(borders_in_cart_sp)
	edge_intensities_expected_stacked = np.vstack(edge_intensities_expected)

	proj_points, _ = cv2.projectPoints( stacked_borders_in_cart_sp [:,0:3], np.zeros((3,1)), np.zeros((3,1)), mtx, dist)
	proj_points_int = np.ndarray.astype(proj_points,int)

	proj_points_int = proj_points_int.reshape(proj_points_int.shape[0],2)
	n_int = proj_points_int.shape[0]
	temp = proj_points.shape[0]
	proj_points = proj_points.reshape(temp,2)
	
	testing_points = temp 
	for i in range(n_int):
		center = tuple(np.ndarray.astype(proj_points_int[i,:],int))
		cv2.circle( frame_gray_draw, center , rad , col_gr, -1)
	


	f_p = frame_gray[proj_points_int[:,1],proj_points_int[:,0]]  # TODO i dont think framegray int16 is needed ? Also 0,1 order changed
	err = (edge_intensities_expected_stacked - f_p.reshape(f_p.shape[0],1))/1.0 # this is the error in the intensities
	return  err.reshape(err.shape[0],)

def LM_DPR_Jacobian(X, frame_gray, ids, corners, b_edge, edge_intensities_expected_all, aruco_images, mtx, dist):

	'''Function to calculate the Jacobian of the objective function'''

	Tf_cam_ball = RodriguesToTransf(X)
	borders_in_cart_sp = []
	edge_intensities_expected = []
	for ii in range(ids.shape[0]):
		Tf_cent_face,Tf_face_cent = tf_mat_dodeca_pen(int(ids[ii]))
		# borders_in_cart_sp.append((Tf_cam_ball.dot(b_edge[ii].T)).T)  ### by arkadeep
		borders_in_cart_sp.append((Tf_cam_ball.dot(Tf_cent_face).dot(b_edge[ii].T)).T)  ### added by tejas

		edge_intensities_expected.append(edge_intensities_expected_all[ii].reshape(edge_intensities_expected_all[ii].shape[0],1))

	stacked_borders_in_cart_sp = np.vstack(borders_in_cart_sp)
	edge_intensities_expected_stacked = np.vstack(edge_intensities_expected)
	proj_points , duvec_by_dp_all = cv2.projectPoints( stacked_borders_in_cart_sp [:,0:3], np.zeros((3,1)), np.zeros((3,1)), mtx, dist)
	proj_points_int = np.ndarray.astype(proj_points,int) # -profile this is taking some time (5.5% of total time)

	proj_points_int = proj_points_int.reshape(proj_points_int.shape[0],2)

	du_by_dp = duvec_by_dp_all[0::2,0:6]
	# dv_by_dp = duvec_by_dp_all[0::1,0:6]  ### by arkadeep
	dv_by_dp = duvec_by_dp_all[1::2,0:6]  ### edited by tejas
 	
	# draw_3d_point(frame,np.zeros((6,),dtype=np.float32),np.array(T_cent_accepted[itr,0:3,3]),(0,255,255),2) 
 
	# dI_by_dv,dI_by_du = local_frame_grads (frame_gray, np.vstack(corners), ids) ##TODO local frame gradients not working pl check ## arkadeep
	# dI_by_dv,dI_by_du = local_frame_grads (frame_gray,X[3:6],300,mtx,dist) ##TODO local frame gradients ## tejas
	
	dI_by_dv,dI_by_du = np.gradient (frame_gray.astype('float32'))  ##  by arkadeep # -profile: 27.88% of time HAVE TO CHANGE

	n_int = proj_points_int.shape[0]
	dI_by_dp = np.zeros((n_int,6))
	for i in range(n_int):
		ui,vi = proj_points_int[i,0], proj_points_int[i,1]
		# dI_by_dp[i,:] = dI_by_du [ui,vi] * du_by_dp[i] + dI_by_dv [ui,vi] * dv_by_dp[i] #TODO confirn [u,v] order in eqn  # by arkadeep
		dI_by_dp[i,:] = dI_by_du [vi,ui] * du_by_dp[i] + dI_by_dv [vi,ui] * dv_by_dp[i] #TODO confirn [u,v] order in eqn  #edited  by tejas

	return -dI_by_dp  # the neg sign is required

######


def main():
	global frame_gray_draw, frame_gray
	img = cv2.imread('send_to_howard.png')
	h,  w = img.shape[:2]

	sub_pix_refinement_switch = 3
	detect_tip_switch = 0
	hist_plot_switch = 1
	show_filter_switch = 1
	show_ape_switch = 0
	run_DPR_switch = 1

	iterations_for_while =5500
	marker_size_in_mm = 17.78
	distance_betn_markers = 34.026  #in mm
	dilate_fac = 1.2 # dilate the square around the marker
	tip_coord  = np.array([3.97135363, -116.99921549 ,-5.32922903,1]) 

	edge_pts_in_img_sp = [0]*13
	aruco_images = [0]*13
	aruco_images_int16 = [0]*13
	img_pnts = [0]*13
	for i in range(1,13):
		edge_pts_in_img_sp[i] = np.loadtxt("thick_edge_coord_R3/id_{}.txt".format(i),delimiter=',',dtype=np.float32)
		aruco_images[i]= cv2.imread("aruco_images_mip_maps/res_150_{}.jpg".format(i),0)
		img_pnts[i] = np.loadtxt("thick_edge_coord_pixels/id_{}.txt".format(i),delimiter=',',dtype='int16')
		aruco_images_int16[i] = np.int16(aruco_images[i])



	with np.load('PTGREY.npz') as X:
		mtx, dist = [X[i] for i in ('mtx','dist')] 
	newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))  ### mtx is newcameramtx


	pose_marker_with_APE= np.zeros((iterations_for_while,6))
	pose_marker_with_DPR= np.zeros((iterations_for_while,6))
	pose_marker_without_opt = np.zeros((iterations_for_while,6))
	#pose_marker_avg = np.zeros((iterations_for_while,6))

	tip_posit = np.zeros((iterations_for_while,3))
	color = [0]*iterations_for_while


	aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
	parameters =  aruco.DetectorParameters_create()
	parameters.cornerRefinementMethod = sub_pix_refinement_switch
	parameters.cornerRefinementMinAccuracy = 0.05
	stkd_2d_corn_pix_sp = np.zeros((12,2))
	markers_possible = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
	markers_impossible = np.array([13,17,37,16,34,45,38,24,47,32,40])
	points_for_DPR = 100

	j = 0  # iteration counter


	time_vect = [0]*iterations_for_while



	cv2.namedWindow('image',cv2.WINDOW_NORMAL)
	rospy.init_node('RosCam', anonymous=True)
	ic = RosCam("/camera/image_color")


	t_prev = time.time()
	while(j<iterations_for_while):
		  
		frame = ic.cv_image

		# print(frame)
		if frame is None:
			time.sleep(0.1)
			print("No image")
			continue
		frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)			
		# mtx = newcameramtx #### ??????


		frame_gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
		frame_gray_draw = np.copy(frame_gray)
		# cv2.imshow('frame_gray',frame)
		# cv2.waitKey(0)
		# frame_gray = frame_gray.astype('float32')/255
		#lists of ids and the corners beloning to each id

		# cv2.imshow('frame_color',frame)

		# the first row will allways be [0,0,0] this is to ensure that we can start from face 1 which is actually face 0
		rvecs = np.zeros((13,1,3))
		tvecs = np.zeros((13,1,3))
		# 
		corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
	 
		if ids not in markers_impossible and ids is not None and len(ids) >= 2: 
			t0 = time.time()
			N_markers =ids.shape[0]
			frame = aruco.drawDetectedMarkers(frame, corners, ids)

			# m is the index for marker ids
			# draw_3d_point(frame,np.zeros((6,),dtype=np.float32),np.array([[0.,0.,200.]])) 
			jj = 0
			# the following are with the camera frame
			cent_in_R3 = np.zeros((N_markers,3))
			T_cent = np.zeros((ids.shape[0],4,4)) 
			for m in ids:
				m_indx = np.asarray(np.where(m==ids))
				rvecs[m,:,:], tvecs[m,:,:], _ = cv2.aruco.estimatePoseSingleMarkers( corners[int(m_indx[0])], marker_size_in_mm, mtx, dist)
				frame = aruco.drawAxis(frame, mtx, dist, rvecs[m,:,:], tvecs[m,:,:], 10)
				T_4_Aruco = RodriguesToTransf(np.append(rvecs[m,:,:], tvecs[m,:,:]))
				T_mat_cent_face,T_mat_face_cent = tf_mat_dodeca_pen(int(m))
				T_cent[jj,:,:] = np.matmul(T_4_Aruco,T_mat_face_cent)
				jj+=1
			

			if show_filter_switch == 1:
				for itr in range(T_cent.shape[0]):	
					draw_3d_point(frame,np.zeros((6,),dtype=np.float32),np.array(T_cent[itr,0:3,3]), mtx, dist) 	

			T_cent_accepted, centers_R3, good_indices = remove_bad_aruco_centers(T_cent, mtx, dist)
			
			if show_filter_switch == 1:	
				for itr in range(T_cent_accepted.shape[0]):
					draw_3d_point(frame,np.zeros((6,),dtype=np.float32),np.array(T_cent_accepted[itr,0:3,3]), mtx, dist, (0,255,255),2) 
				
			# print T_cent_accepted," T_cent_accepted.shape"
			# print good_indices,"good_indices"
			# print centers_R3,"centers_R3"


			# Tf_cam_ball = np.mean(T_cent_accepted,axis=0)   ########## by arkadeep

			Tf_cam_ball= find_tfmat_avg (T_cent_accepted) ########## by tejas


			# getting the rvecs and t vecs by averaging
			r_vec_aruco,_ = cv2.Rodrigues(Tf_cam_ball[0:3,0:3])
			t_vec_aruco = Tf_cam_ball[0:3,3]
			stacked_corners_px_sp =  np.reshape(np.asarray(corners),(ids.shape[0]*4,2))
			
			X_guess = np.append(r_vec_aruco,np.reshape(t_vec_aruco,(3,1))).reshape(6,1)
			pose_marker_without_opt[j,:] = X_guess.T # not efficient. May have to change


			t0 = time.time()
			res = leastsq (LM_APE_Dodecapen,X_guess,Dfun=None, full_output=0, 
				col_deriv=0, ftol=1.49012e-6, xtol=1.49012e-4, gtol=0.0, 
				maxfev=1000, epsfcn=None, factor=1, diag=None, 
				args = (stacked_corners_px_sp, ids, marker_size_in_mm, mtx, dist, False)) 
			
			pose_marker_with_APE[j,:] = np.reshape(res[0],(1,6))
			Tf_cam_ball = RodriguesToTransf(res[0])

			# sq_len = 35.
			# draw_3d_point(frame_gray_draw,np.zeros((6,),dtype=np.float32),
			# 		np.array(T_cent_accepted[itr,0:3,3]+[sq_len,sq_len,0]),(0,255,255),2) 
			# draw_3d_point(frame_gray_draw,np.zeros((6,),dtype=np.float32),
			# 		np.array(T_cent_accepted[itr,0:3,3]+[-sq_len,sq_len,0]),(0,255,255),2) 
			# draw_3d_point(frame_gray_draw,np.zeros((6,),dtype=np.float32),
			# 		np.array(T_cent_accepted[itr,0:3,3]+[-sq_len,-sq_len,0]),(0,255,255),2) 
			# draw_3d_point(frame_gray_draw,np.zeros((6,),dtype=np.float32),
			# 		np.array(T_cent_accepted[itr,0:3,3]+[sq_len,-sq_len,0]),(0,255,255),2) 


	###############################################################################################################################
	################# this part is for projecting the corner points on the
	################# frame
	#################
			# px_sp,_ = cv2.projectPoints(np.reshape(res[0][3:6],(3,1)).T, np.zeros((3,1)), np.zeros((3,1)), mtx, dist)
			# temp1 = int(px_sp[0,0,0])
			# temp2 = int(px_sp[0,0,1])
			# if show_ape_switch == 1:
			# 	cv2.circle(frame,(temp1,temp2), 8 , (0,0,255), 3)
			# no_of_accepted_points = len(good_indices) 
						 
			# if no_of_accepted_points is not 0:
			# 	px_sp,_ = cv2.projectPoints(np.reshape(t_vec_aruco,(3,1)).T, np.zeros((3,1)), np.zeros((3,1)), mtx, dist)
			# 	temp1 = int(px_sp[0,0,0])
			# 	temp2 = int(px_sp[0,0,1])
			# 	# cv2.circle(frame,(temp1,temp2), 10 , (0,255,0), 2)

			# N_corners = ids.shape[0]
			
			
			# corners_in_cart_sp = np.zeros((ids.shape[0],4,3))
			
			# for ii in range(ids.shape[0]):
			# 	Tf_cent_face,Tf_face_cent = tf_mat_dodeca_pen(int(ids[ii]))
			# 	corners_in_cart_sp[ii,:,:] = Tf_cam_ball.dot(corners_3d(Tf_cent_face,marker_size_in_mm)).T[:,0:3]
			
			# corners_in_cart_sp = corners_in_cart_sp.reshape(ids.shape[0]*4,3)
			# projected_in_pix_sp,_ = cv2.projectPoints(corners_in_cart_sp,np.zeros((3,1)),np.zeros((3,1)),mtx,dist) 
			# projected_in_pix_sp = projected_in_pix_sp.reshape(projected_in_pix_sp.shape[0],2)
			
			# projected_in_pix_sp_int = np.int16(projected_in_pix_sp)
			# print Tf_cam_ball,"after APE"
			# for iii in range(projected_in_pix_sp_int.shape[0]):
			# 	cv2.circle(frame,(projected_in_pix_sp_int[iii,0],projected_in_pix_sp_int[iii,1]), 5 , (0,255,0),-1)
			# 	cv2.circle(frame,(stacked_corners_px_sp[iii,0],stacked_corners_px_sp[iii,1]), 3 , (255,0,0),-1)
	###############################################################################################################################
			


		# test for DPR
			# print(np.vstack(corners).shape,'np.asarray(corners)')
			# frame_grad_u,frame_grad_v = local_frame_grads (frame_gray, np.vstack(corners), ids)
			# cv2.imshow('frame_grad_u',frame_grad_v)
			# cv2.imshow("DPR frame", frame_gray)
			# cv2.waitKey(0)

			b_edge, edge_intensities_expected =  marker_edges(ids,points_for_DPR,edge_pts_in_img_sp,aruco_images_int16,img_pnts)
	 
			# res_DPR = leastsq (LM_DPR,res[0], Dfun= LM_DPR_Jacobian, full_output=0, #### by arkadeep   
			# 	col_deriv=0, ftol=1.49012e-8, xtol=1.49012e-8, gtol=0.0, 
			# 	maxfev=1000, epsfcn=None, factor=1, diag=None, 
			# 	args = (frame_gray, ids, corners, b_edge, edge_intensities_expected,aruco_images) ) 
			
			if run_DPR_switch == 1:
				# LM_DPR_DRAW(res[0], frame_gray_draw, ids, corners,            # drawing points as result of APE 
				# 			b_edge, edge_intensities_expected,aruco_images, mtx, dist, 250,2)

				res_DPR = leastsq (LM_DPR, res[0], Dfun= LM_DPR_Jacobian,full_output=1,   #### by Tejas 
					col_deriv=0, ftol=1.49012e-6, xtol=1.49012e-4, gtol=0.0, 
					maxfev=1000, epsfcn=None, factor=1, diag=None,
					args = (frame_gray, ids, corners, b_edge, edge_intensities_expected, aruco_images, mtx, dist) ) 

				# LM_DPR_DRAW(res_DPR[0], frame_gray_draw, ids, corners,        # drawing points as result of DPR
				# 			 b_edge, edge_intensities_expected,aruco_images, mtx, dist, 0,1)

			else :
				res_DPR = res

			print res_DPR[2]["nfev"],"res_DPR[2].nfev"	
			pose_marker_with_DPR[j,:] = np.reshape(res_DPR[0],(1,6))

			Tf_cam_ball = RodriguesToTransf(res_DPR[0])
	 
	 
			px_sp,_ = cv2.projectPoints(np.reshape(res_DPR[0][3:6],(3,1)).T, np.zeros((3,1)), np.zeros((3,1)), mtx, dist)
			temp1 = int(px_sp[0,0,0])
			temp2 = int(px_sp[0,0,1])
			cv2.circle(frame,(temp1,temp2), 5 , (0,255,255), 2)
	 
			# print(np.linalg.norm(res[0] - res_DPR[0], 2),"DPR Improvement")

			# angle_range = np.arange(-0.5,0.5,0.01)
			# tra_range = angle_range*10
			# obj_func_val = np.zeros((tra_range.shape[0],6))
			# for ii in range(tra_range.shape[0]):
			# 	obj_func_val[ii,0] = LM_DPR(res[0] + np.array([angle_range[ii],0.,0.,0.,0.,0.]), frame_gray, ids,  b_edge, edge_intensities_expected,aruco_images).sum()
			# 	obj_func_val[ii,1] = LM_DPR(res[0] + np.array([0.,angle_range[ii],0.,0.,0.,0.]), frame_gray, ids,  b_edge, edge_intensities_expected,aruco_images).sum()
			# 	obj_func_val[ii,2] = LM_DPR(res[0] + np.array([0.,0.,angle_range[ii],0.,0.,0.]), frame_gray, ids,  b_edge, edge_intensities_expected,aruco_images).sum()
			# 	obj_func_val[ii,3] = LM_DPR(res[0] + np.array([0.,0.,0.,tra_range[ii],0.,0.]), frame_gray, ids,  b_edge, edge_intensities_expected,aruco_images).sum()
			# 	obj_func_val[ii,4] = LM_DPR(res[0] + np.array([0.,0.,0.,0.,tra_range[ii],0.]), frame_gray, ids,  b_edge, edge_intensities_expected,aruco_images).sum()
			# 	obj_func_val[ii,5] = LM_DPR(res[0] + np.array([0.,0.,0.,0.,0.,tra_range[ii]]), frame_gray, ids,  b_edge, edge_intensities_expected,aruco_images).sum()
			# fig1 = plt.figure()
			# plt.scatter(angle_range[:],obj_func_val[:,0],s = 5,label = "ang0")
			# plt.scatter(angle_range[:],obj_func_val[:,1],s = 5,label = "ang1")
			# plt.scatter(angle_range[:],obj_func_val[:,2],s = 5,label = "ang2")
			# plt.legend()
			# plt.show()

			# fig1 = plt.figure()
			# plt.scatter(tra_range[:],obj_func_val[:,3],s = 5,label = "tra0")
			# plt.scatter(tra_range[:],obj_func_val[:,4],s = 5,label = "tra1")
			# plt.scatter(tra_range[:],obj_func_val[:,5],s = 5,label = "tra2")
			# plt.legend()
			# plt.show()

			t_new = time.time() 
			print("current frame rate",1./(t_new - t_prev))

			t_prev = t_new
			j = j+1
			print("frame number ", j)
			cv2.imshow('frame_gray_draw',frame_gray_draw)
			# cv2.imshow('frame_gray',frame_gray)
			# cv2.imshow('aruco_images',aruco_images[ids[0]])
			# cv2.imshow('frame_color',frame)
			if cv2.waitKey(1) & 0xFF == ord('q') or j >= iterations_for_while:
				break
		else: 
			print("Required marker not visible")
			# cv2.imshow('frame',frame)
			if cv2.waitKey(1) & 0xFF == ord('q') or j >= iterations_for_while:
				break
		


	cv2.destroyAllWindows()



	#### Analysis


	pose_marker_with_APE = pose_marker_with_APE[0:j,:]

	pose_marker_with_DPR= pose_marker_with_DPR[0:j,:]
	pose_marker_without_opt = pose_marker_without_opt[0:j,:]
	#pose_marker_avg = pose_marker_avg[0:j,:]
	tip_posit = tip_posit[0:j,:]



	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")

	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	if detect_tip_switch == 1:
	#    print (np.std(tip_posit,axis=0), "std in x,y,z")q
		ax.scatter(tip_posit[:,0],tip_posit[:,1],tip_posit[:,2],c=color)
	else: 
		print ("the end")
		ax.scatter(pose_marker_without_opt[:,3],pose_marker_without_opt[:,4],pose_marker_without_opt[:,5],c ='m',label = "pose_marker_without_opt")

		# fig = plt.figure()
		# ax = fig.add_subplot(111, projection="3d")

		ax.scatter(pose_marker_with_APE[:,3],pose_marker_with_APE[:,4],pose_marker_with_APE[:,5],c = 'r',label="pose_marker_with_APE" )
		ax.scatter(pose_marker_with_DPR[:,3],pose_marker_with_DPR[:,4],pose_marker_with_DPR[:,5],c = 'g',label="pose_marker_with_DPR" )
		ax.legend()
		
		# plt.axis('equal')
	if hist_plot_switch == 1:
		fig = plt.figure()
		plt.hist(pose_marker_without_opt[:,5],j,facecolor='magenta',normed = 1,label = 'pose_marker_without_opt' )
		# fig = plt.figure()
		plt.hist(pose_marker_with_APE[:,5],j,facecolor='red',normed = 1, label = 'pose_marker_with_APE'  )
		plt.hist(pose_marker_with_DPR[:,5],j,facecolor='green',normed = 1, label = 'pose_marker_with_DPR'  )
		plt.legend()

	plt.show()

if __name__ == '__main__':
	main()