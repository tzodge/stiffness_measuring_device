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
#from helper import *
from scipy.optimize import minimize, leastsq
from scipy import linalg
from scipy.interpolate import griddata
#import numdifftools as nd

### functions ##
###Drawing functions 
def draw_marker_edges(frame,pose_obj,id_no,color=(0,0,255)):
	## for now, the ids are assigned sloppily, later add a dictionary containing marker transformations
	if id_no == 7:
		b = b7_wrt_obj
	elif id_no == 8:
		b = b8_wrt_obj
	elif id_no == 9:
		b = b9_wrt_obj
	points_to_project, Jac  = cv2.projectPoints(b[0:3,:].T,pose_obj[0:3],pose_obj[3:6],mtx,dist)
	
####### following lines for testing
	### verifies that for marker number 8, the translation part 
	### of the jacobian is what same as what we compute analytically
	### verifies this at the last point of the 3d transformed points

	# print b[0:3,:].shape, "b[0:3,:].shape"
	# print Jac.shape, "Jac"
	# print Jac[:,0:6], "Jac[:,0:6]"
	# b8 = tf_c_8.dot(np.eye(4)).dot(b8_native)   
	# print partial_u_wrt_x_hat(mtx,b8[0:3,-1]), "partial_u_wrt_x_hat"

	# print ""

	points_to_project = points_to_project.reshape(points_to_project.shape[0],2)
 
	points_to_project = np.fliplr(points_to_project.T)
	
	for i in range(points_to_project.shape[1]):
		cv2.circle( frame, tuple(np.ndarray.astype(points_to_project[:,i],int)) , 1 , color, -1)
	# cv2.imshow("image_draw",frame)
	# cv2.waitKey(0)
	
	



def objective_fun_LM(x):
	tf_c_8 = RodriguesToTransf(x)
	corners_for_8 = tf_c_8.dot(corners_3d(np.eye(4),marker_size_in_mm))
	corners_for_7 = tf_c_8.dot(corners_3d(tf_8_7,marker_size_in_mm))
	corners_for_9 = tf_c_8.dot(corners_3d(tf_8_9,marker_size_in_mm))
	corners_in_cart_sp = np.hstack((corners_for_7,corners_for_8,corners_for_9))
	corners_in_cart_sp = corners_in_cart_sp[0:3,:].T
	projected_in_pix_sp,_ = cv2.projectPoints(corners_in_cart_sp,np.zeros((3,1)),np.zeros((3,1)),mtx,dist) 
	projected_in_pix_sp = projected_in_pix_sp.reshape(12,2)
	
	A = projected_in_pix_sp
	B = stkd_2d_corn_pix_sp
	n,_=np.shape(A)
	V= LA.norm(A-B, axis=1)
	return V/n



def objective_fun_NM(x):
	tf_c_8 = RodriguesToTransf(x)
	corners_for_8 = tf_c_8.dot(corners_3d(np.eye(4),marker_size_in_mm))
	corners_for_7 = tf_c_8.dot(corners_3d(tf_8_7,marker_size_in_mm))
	corners_for_9 = tf_c_8.dot(corners_3d(tf_8_9,marker_size_in_mm))
	corners_in_cart_sp = np.hstack((corners_for_7,corners_for_8,corners_for_9))
	corners_in_cart_sp = corners_in_cart_sp[0:3,:].T
	projected_in_pix_sp,_ = cv2.projectPoints(corners_in_cart_sp,np.zeros((3,1)),np.zeros((3,1)),mtx,dist) 
	projected_in_pix_sp = projected_in_pix_sp.reshape(12,2)
	
	A = projected_in_pix_sp
	B = stkd_2d_corn_pix_sp
	n,_=np.shape(A)
	V= LA.norm(A-B, axis=1)
	return V.sum()/n


def objective_fun_LM_DPR(x):
	
	tf_c_8 = RodriguesToTransf(x)
	
	b7 = marker_edges(7,4000)
	b8 = marker_edges(8,4000)
	b9 = marker_edges(9,4000)
	
	tf_8_7, tf_8_9 = tf_mat_pla_pen (distance_betn_markers)
	
	b7 =  tf_c_8.dot(tf_8_7).dot(b7)
	b8 = tf_c_8.dot(np.eye(4)).dot(b8)
	b9 =  tf_c_8.dot(tf_8_9).dot(b9)
	
	b = np.hstack((b7,b8,b9))
	
	points_in_R3 = b[0:3,:].T
	
	
	transf_and_proj,_ = cv2.projectPoints(points_in_R3, np.zeros((3,1)),np.zeros((3,1)),mtx,dist)
	transf_and_proj = transf_and_proj.reshape(transf_and_proj.shape[0],2)
	transf_and_proj_int = np.ndarray.astype(np.ndarray.round( transf_and_proj),int)

	Ic_p =  frame[transf_and_proj_int[:,1],transf_and_proj_int[:,0]]
	
	V = Ic_p
	n = V.shape[0]
	return V/n

def tf_mat_pla_pen (m_d):
	transf8_9 = np.array(   [[1, 0, 0, 0],\
							[0,1,0,+m_d], \
							[0,0,1,0],     \
							[0,0,0,1]]) 

	transf8_7 = np.array(   [[1, 0, 0, 0],\
							[0,1,0,-m_d], \
							[0,0,1,0],     \
							[0,0,0,1]]) 

	return transf8_7, transf8_9

def corners_3d(tf_mat,m_s):
	corn_1 = np.array([-m_s/2.0,  m_s/2.0, 0, 1])
	corn_2 = np.array([ m_s/2.0,  m_s/2.0, 0, 1])
	corn_3 = np.array([ m_s/2.0, -m_s/2.0, 0, 1])
	corn_4 = np.array([-m_s/2.0, -m_s/2.0, 0, 1])

	corn_mf = np.vstack((corn_1,corn_2,corn_3,corn_4))
	corn_pgn_f = tf_mat.dot(corn_mf.T)

	return corn_pgn_f


def marker_edges (marker_id, dwn_smpl):
	
	marker_id_str = "aruco_images" + "/{}.jpg".format(marker_id)  
	gray = cv2.imread(marker_id_str)
	a = cv2.Canny(gray,100,200,apertureSize = 5)
	a[0,:] = 255
	a[-1,:] = 255
	a[:,0] = 255
	a[:,-1] = 255
	# cv2.imshow("image",a)
	a = np.fliplr(a.T) #do not change this!!! #            
	scale = a.shape[0]/marker_size_in_mm
	a_shift = a.shape[1]/(2*scale)
	b = np.asarray(np.nonzero(a))/scale - a_shift
	z = np.zeros((1,b.shape[1]))
	tr_dash = np.ones((1,b.shape[1]))
	b = np.vstack ((b,z,tr_dash))       
	interval =  int(b.shape[1]/dwn_smpl)
#    print (interval,"interval " )
	b = b[:,0::interval]
	return b

def get_marker_borders (corners,dilate_fac):

	cent = np.array([np.mean(corners[:,0]), np.mean(corners[:,1])])
	
	vert_1 = (corners[0,:] - cent)* dilate_fac
	vert_2 = (corners[1,:] - cent)* dilate_fac
	vert_3 = (corners[2,:] - cent)* dilate_fac
	vert_4 = (corners[3,:] - cent)* dilate_fac
	
	expanded_corners = np.vstack((vert_1+cent,vert_2+cent,vert_3+cent,vert_4+cent))
	 
	return expanded_corners

def normalize_markers (frame,ids,stacked_corners,dilate_fac,thresh_low,thresh_high):
	# this will normalize around the markers, threshold it and will also return a 
	#different frame with the gradients of the frame near the markers
	n_markers = ids.shape[0]
	frame_grad_u = np.zeros(frame.shape[0:2])
	frame_grad_v = np.zeros(frame.shape[0:2])

	##### frame_grad_u_img, frame_grad_v_img is just to be used for plotting. 
	##### this is done because the image object can't store negative values and we need negative 
	##### values of gradients
	frame_grad_u_img = np.copy(frame)
	frame_grad_v_img = np.copy(frame)
	frame_gray_norm = np.copy(frame)

	# initialize required frames
	for i in range(n_markers):
		expanded_corners = get_marker_borders(stacked_corners[i,:,:],dilate_fac)
		
		### do not change order of indices
		x_low = int(np.min(expanded_corners[:,1]))
		x_high = int(np.max(expanded_corners[:,1])) 
		y_low = int(np.min(expanded_corners[:,0]))
		y_high = int(np.max(expanded_corners[:,0]))
	   ### do not change order of indices
	   
		frame_norm_cropped = frame_gray[x_low:x_high,y_low:y_high]
		frame_norm_cropped = cv2.normalize(frame_norm_cropped, dist, thresh_low, thresh_high, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		frame_gray_norm[x_low:x_high,y_low:y_high] = frame_norm_cropped
		
		A = cv2.Sobel(frame_gray_norm[x_low:x_high,y_low:y_high],cv2.CV_64F,1,0,ksize=5)
		B = cv2.Sobel(frame_gray_norm[x_low:x_high,y_low:y_high],cv2.CV_64F,0,1,ksize=5)
		
		frame_grad_u[x_low:x_high,y_low:y_high] = A
		frame_grad_v[x_low:x_high,y_low:y_high] = B

		frame_grad_u_img[x_low:x_high,y_low:y_high] = A
		frame_grad_v_img[x_low:x_high,y_low:y_high] = B


###########  to verify that if we set the frame_grad_u as an image, all of it's components are positive 
		# frame_grad_u = frame_grad_u_img
		# frame_grad_v = frame_grad_v_img


	return frame_gray_norm, frame_grad_u, frame_grad_v, frame_grad_u_img, frame_grad_v_img

def skew_matrix(v):
	M = np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1], v[0], 0]])
	return(M)
	
def RodriguesToTransf(x):
	#input (6,) (rvec,tvec)
	x = np.array(x)
	rot,_ = cv2.Rodrigues(x[0:3])
	trans =  np.reshape(x[3:6],(3,1))
	Trransf = np.concatenate((rot,trans),axis = 1)
	Trransf = np.concatenate((Trransf,np.array([[0,0,0,1]])),axis = 0)
	return Trransf
	
	
def partial_R_wrt_r (r):
	
	R,_ = cv2.Rodrigues(r)

	norm_r = LA.norm(r,2)
	skw_r = skew_matrix(r)
	
	t_x = np.matmul(np.eye(3)-R, np.array([1,0,0]))
	t_y = np.matmul(np.eye(3)-R, np.array([0,1,0]))
	t_z = np.matmul(np.eye(3)-R, np.array([0,0,1]))
	
	del_R_by_del_r_x = (r[0]*skw_r + skew_matrix(np.cross(r,t_x)))/norm_r**2 *(R)
	del_R_by_del_r_y = (r[1]*skw_r + skew_matrix(np.cross(r,t_y)))/norm_r**2 *(R)
	del_R_by_del_r_z = (r[2]*skw_r + skew_matrix(np.cross(r,t_z)))/norm_r**2 *(R)
	
	del_R_del_r = np.vstack((del_R_by_del_r_x,del_R_by_del_r_y,del_R_by_del_r_z))

	return del_R_del_r

def partial_x_hat_wrt_R_hat (coords_wrt_obj):
	del_x_hat_del_R_hat = np.array([[coords_wrt_obj[0],coords_wrt_obj[1],coords_wrt_obj[2],0,0,0,0,0,0],[0,0,0,coords_wrt_obj[0],coords_wrt_obj[1],coords_wrt_obj[2],0,0,0],[0,0,0,0,0,0,coords_wrt_obj[0],coords_wrt_obj[1],coords_wrt_obj[2]]])
	return del_x_hat_del_R_hat

def partial_u_wrt_x_hat(mtx,tvecs):
	fx = mtx[0,0]
	fy = mtx[1,1]
	del_u_del_x_hat = np.array([[fx/tvecs[2], 0, -fx*tvecs[0]/(tvecs[2]**2)],[0, fy/tvecs[2], -fy*tvecs[1]/(tvecs[2]**2) ] ]) 
	return del_u_del_x_hat 
		
def partial_Ic_wrt_pix(frame_grad_u, frame_grad_v,u,v):
	del_Ic_del_u  = np.array([0,0])
	
	del_Ic_del_u [0] = frame_grad_u[u,v]
	del_Ic_del_u [1] = frame_grad_v[u,v]
	
	return del_Ic_del_u 

def Intensity_Jacobian (rvecs,tvecs,mtx,frame_grad_u, frame_grad_v,transf_and_proj_int):
	R,_ = cv2.Rodrigues(rvecs) 
	N = transf_and_proj_int.shape[0]
	Jac = np.zeros((N,6))
	
	b = np.hstack((b7_wrt_obj,b8_wrt_obj,b9_wrt_obj))
	for i in range(N):
		
		
		u,v = transf_and_proj_int[i,0],transf_and_proj_int[i,1]
		del_R_del_r = partial_R_wrt_r (rvecs)
		del_x_hat_del_R_hat = partial_x_hat_wrt_R_hat (b[:,i])
		del_u_del_x_hat  = partial_u_wrt_x_hat(mtx,tvecs)
		del_Ic_del_u  = partial_Ic_wrt_pix(frame_grad_u, frame_grad_v,u,v)
#        cv2.imshow("frame_grad_u",frame_grad_u)
#        cv2.imshow("frame_grad_v",frame_grad_v)
#        print (frame_grad_v,"frame_grad_v")
#        cv2.waitKey(0)
#        print (del_Ic_del_u  ,"del_Ic_del_u  ")
		
		term_1 = del_u_del_x_hat.dot(del_x_hat_del_R_hat).dot(del_R_del_r)
		
		term_2 = np.hstack((term_1,del_u_del_x_hat))
		
		Jac[i,:] = (del_Ic_del_u.dot(term_2)).reshape(6,)
		
#    J_ps_inv = LA.solve(((Jac.T).dot(Jac)), np.eye(6)).dot(Jac.T)
	# Q,R = LA.qr(Jac)
	# J_ps_inv = LA.inv(R).dot(Q.T)          
#    cond_no =  LA.cond(J_ps_inv)
	# print(Q,"q.shape")
	# print(R,"r.shape")
	# lambda_reg = 0.5
	J_sq = Jac.T.dot(Jac)
	Jac_reg = J_sq + np.diag(J_sq)*np.eye(6)

	return Jac,Jac_reg
 
	 
   
def error_DPR(p): 
	# print(p,"P in error")
	global iii
	tf_c_8 = RodriguesToTransf(p)
	
	tf_8_7, tf_8_9 = tf_mat_pla_pen (distance_betn_markers)
	
	b7 =  tf_c_8.dot(tf_8_7).dot(b7_native)
	b8 = tf_c_8.dot(np.eye(4)).dot(b8_native)
	b9 =  tf_c_8.dot(tf_8_9).dot(b9_native)
	
	b = np.hstack((b7,b8,b9))
	
	points_in_R3 = b[0:3,:].T
	
	
	transf_and_proj,_ = cv2.projectPoints(points_in_R3, np.zeros((3,1)),np.zeros((3,1)),mtx,dist)
	transf_and_proj = transf_and_proj.reshape(transf_and_proj.shape[0],2)
	transf_and_proj_int = np.ndarray.astype(np.ndarray.round( transf_and_proj),int)
	Ic_p =  frame[transf_and_proj_int[:,0],transf_and_proj_int[:,1]]

	Ic_p = cv2.normalize(Ic_p, dist, thresh_low, thresh_high, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

	# n = Ic_p.shape[0]
	# print (Ic_p,"Ic_p")

	for i in range(transf_and_proj.shape[0]):
		cv2.circle( frame_color, tuple(np.ndarray.astype(transf_and_proj[i,:],int)) , 1 , (0,30*iii,0), -1)
#    
	iii += 1
	
	return LA.norm(Ic_p,1), Ic_p, transf_and_proj_int

def find_Jac_and_err (pose_obj, frame_grad_u, frame_grad_v):
   ### takes image, and Approximate pose as input and returns 6xN Jac_dIc_by_dp
	pix_coord, du_by_dp = cv2.projectPoints(all_points_wrt_obj[0:3,:].T,pose_obj[0:3],pose_obj[3:6] ,mtx ,dist)
	## du_by_dp = nx6
	du_by_dp = du_by_dp[:,0:6]
	n = all_points_wrt_obj.shape[1]
	print n, "n"
	pix_coord = np.ndarray.astype(np.ndarray.round( pix_coord),int)
	Jac_dIc_by_dp = np.zeros((n,6))
	err = np.zeros((n,1))
	print pix_coord.shape, "pix_coord.shape"
	for i in range(n):
		# u,v = pix_coord[i,0], pix_coord[i,1]
		dIc_by_dui = frame_grad_u[pix_coord[i,0][1],pix_coord[i,0][0]]
		dIc_by_dvi = frame_grad_v[pix_coord[i,0][1],pix_coord[i,0][0]]
		
		# print dIc_by_dui,dIc_by_dvi, "dIc_by_dui,dIc_by_dvi"
		# print frame_gray[pix_coord[i,0][1],pix_coord[i,0][0]], "frame_grad_u[pix_coord[i,0][1],pix_coord[i,0][0]]" 
		# print frame_gray[pix_coord[i,0][1]+1,pix_coord[i,0][0]], "frame_grad_u[pix_coord[i,0][1]+1,pix_coord[i,0][0]]" 
		
		### finding nth component of the jacobian = [dIc_by_dui,dIc_by_dui].[dui_by_dp,dvi_by_dp]'  = 1x2 . 2x6 = 1x6

		Jac_dIc_by_dp[i,:] = dIc_by_dui*(du_by_dp[2*i,:]) + dIc_by_dvi*( du_by_dp[2*i+1,:])         
		err[i] = ((frame_gray[pix_coord[i,0][1],pix_coord[i,0][0]] - 125 )**2)/n
		# print err[i], "err[i]"
	#### can be used to check the validity of the matrix    
	# print np.linalg.cond(Jac_dIc_by_dp)
	print err.sum(),"err.sum()" 
	
	return Jac_dIc_by_dp, err 



def gauss_newton(p):
	
	del_p_tol = 1e-8
	del_Err_tol = 1e-8
	ii = 0  
	
	p_GN = np.zeros((1000,6))
	p_GN [0,:] = p
#    print (p,"p inside GN")
	Err_GN = np.zeros((1000,1))
	del_Err_GN = np.zeros((1000,1))
	del_p_mag = np.zeros((1000,1))
	del_p_mag[0,0] = 1e+6
	del_Err_GN[0,0] = 1e+6
	  
	while (del_p_mag[ii,0] > del_p_tol) or (del_Err_GN[ii,0] > del_Err_tol ):
		
		#### del_p = Jac_ps_inv*(I_t - I_c)
		#### we need Jac_ps_inv 
		#### for jac_ps_inv, we need transf_and_proj_int 
		Err_GN[ii,0], It_m_Ic, transf_and_proj_int = error_DPR(p_GN[ii,:])
		_, J_ps_inv = Intensity_Jacobian (p_GN[ii,0:3],p_GN[ii,3:6],mtx,frame_grad_u, frame_grad_v, transf_and_proj_int)
		
#        J_ps_inv = find_Jac_ps_inv (frame_gray,pose)
		
		
		p_GN[ii+1,:] = p_GN[ii,:] - J_ps_inv.dot(It_m_Ic)
		Err_GN[ii+1,0], _, _ = error_DPR(p_GN[ii+1,:])
		
		del_Err_GN[ii+1,0] = Err_GN[ii+1,0] - Err_GN[ii,0]                  
		del_p_mag[ii+1,0] = LA.norm((p_GN[ii+1,:] - p_GN[ii,:]),1)
		
#        print (p_GN[ii,:],"p_GN[{},0]".format(ii))
#        print (p_GN[ii+1,:],"p_GN[{},0]".format(ii+1))
		print(del_Err_GN[ii,0],"del_Err_GN[ii,0]")
		print (del_p_mag[ii,0],"del_p_mag[ii,0]")
		print (ii, "ii, inside GN")
		# draw_marker_edges(frame_color,p_GN[ii,:],8)
		ii += 1
		
	
	p_GN = p_GN [0:ii,:]

	return p_GN[-1,:]
	
# def gauss_newton(p):
#     print(p,'p in gauss newton')
#     change_in_p = 1e+6
#     iter_1 = 0
#     E_new_sc = np.inf
#     del_p = np.zeros((6,))
#     while change_in_p > gauss_newt_tol and iter_1<200:
#         iter_1 += 1
#         E_curr_sc,E_curr_vec,transf_and_proj_int = error_DPR(p+del_p)
#         # grad_E_curr = nd.Gradient(error_DPR)(np.r_[p_str[0], p_str[1],p_str[2],p_str[3],p_str[4],p_str[5]])
		
# #         _, Q,R = Intensity_Jacobian (p[0:3],p[3:6],mtx,frame_grad_u, frame_grad_v, transf_and_proj_int)
# #         # del_p = J_ps_inv.dot(E_curr_vec)
# #         # _, Q,R = Intensity_Jacobian (p[0:3],p[3:6],mtx,frame_grad_u, frame_grad_v, transf_and_proj_int)
# # #        print(Q.shape)
# # #        print(R.shape)
# #         c = Q.T.dot(E_curr_vec)
#         print(iter_1,"iter_1")
#         Jac,Jac_reg = Intensity_Jacobian (p[0:3],p[3:6],mtx,frame_grad_u, frame_grad_v, transf_and_proj_int)
#         del_p = linalg.solve_triangular(Jac_reg,-Jac.T.dot(E_curr_vec))
#         print(del_p,"del_p")
		
#         # print(c.shape)
#         # del_p = linalg.solve_triangular(R,c)
#         # print(LA.cond(R),"condition_of_R")
		
#         del_p = np.reshape(del_p,(6,))
#         # print(del_p,"del_p")
#         # draw_marker_edges(frame_color,p,8)
#         p = p + del_p
#         # print(p,"p_old")
#         # E_new_sc,_,_ =  error_DPR(p)
#         # print (E_new_sc,"E_new_sc" )
#         # print (E_curr_sc,"E_curr_sc")
#         j = 1
#         E_new_sc = np.inf
#         while (E_new_sc > E_curr_sc) and j<50:
#             del_p = alpha*del_p
#             p_str = p + del_p
#             E_new_sc,_,_ = error_DPR(p_str)            
#             # print (j,"j") 
#             j = j+1
		   
#         change_in_p = LA.norm(p-p_str,1) 
#         print(change_in_p,"error")
#         print(E_curr_sc,"E_curr_sc")

#     return p   

######

def draw_synth_img(frame,pose,id_no):
	frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	frame_draw_white = np.copy(frame_gray)
	frame_draw_white[:,:] = 255 
	## load aruco image m x n matrix
	marker_id_str = "aruco_images" + "/{}.jpg".format(id_no)  
	aruco_img = cv2.imread(marker_id_str)
	aruco_img[100:110,200:210] = 0
	cv2.imshow("aruco_img",aruco_img)
	aruco_img_gray = cv2.cvtColor(aruco_img,cv2.COLOR_BGR2GRAY)
	aruco_img_gray = np.fliplr(aruco_img_gray.T)

	n = aruco_img.shape[0]
	Int_val = aruco_img_gray.reshape(n**2,1)
	print aruco_img_gray.shape,"aruco_img_gray.shape"
	# n = 100
	scale_fact = marker_size_in_mm/n
	# scale_fact = 1
	img_points_3d = np.zeros((4,n**2))
	img_points_3d[3,:] = 1
	for i in range(n):
		# print i
		img_points_3d[0,n*i:n*(i+1)] = (i-n/2.0) *scale_fact
		img_points_3d[1,n*i:n*(i+1)] = (np.arange(0,n)-n/2.0)*scale_fact
	tf_mat = RodriguesToTransf(pose.reshape(6,))	
	transformed_points = tf_mat.dot(img_points_3d)
	# print transformed_points[:,0:5]," transformed_points[:,0:3]"
	proj_points,_ = cv2.projectPoints(transformed_points[0:3,:].T,np.zeros((1,3)),np.zeros((1,3)),mtx,dist)


	proj_points_int = np.ndarray.astype(proj_points,int)
	n1 = proj_points_int.shape[0]
	proj_points_int = proj_points_int.reshape(n1,2)
	proj_points_int = np.unique(proj_points_int,axis=0)
	print proj_points_int.shape, "proj_points_int.shape"

	proj_points = proj_points.reshape(n*n,2)

	Int_val_interpolated = griddata(proj_points,Int_val,(proj_points_int[:,0],proj_points_int[:,1]), method = 'cubic')
	
	# frame_gray[100:105,200:205] = 0

	# cv2.circle( frame_gray, tuple(np.ndarray.astype(np.array([100.0,200.0]),int)) , 10 , (255,255,255), -1)


	for i in range(proj_points_int.shape[0]):
		cv2.circle( frame_draw_white, tuple(np.ndarray.astype(proj_points_int[i,:],int)) , 1 , Int_val_interpolated[i], -1)
		cv2.circle( frame_gray, tuple(np.ndarray.astype(proj_points_int[i,:],int)) , 1 , Int_val_interpolated[i], -1)


	# points_to_project, Jac  = cv2.projectPoints(b[0:3,:].T,pose_obj[0:3],pose_obj[3:6],mtx,dist)

	cv2.imshow("frame_gray_synth",frame_draw_white)
	cv2.imshow("frame_gray",frame_gray)
	cv2.waitKey(0)





### Switches: 
global transf_and_proj_int 

iii = 0

sub_pix_refinement_switch =1
detect_tip_switch = 0
hist_plot_switch = 1


iterations_for_while = 2000
marker_size_in_mm = 17.78
distance_betn_markers = 34.026  #in mm
dilate_fac = 1.5 # dilate the square around the marker
thresh_low = 0
thresh_high = 10
tip_coord  = np.array([3.97135363, -116.99921549 ,-5.32922903,1]) 
gauss_newt_tol = 1e-8
alpha = 0.5
c = 1e-4
pts_per_marker_sampled_DPR = 60


tf_8_7, tf_8_9 = tf_mat_pla_pen (distance_betn_markers)

b7_native =  marker_edges(7,pts_per_marker_sampled_DPR)
b8_native =  marker_edges(8,pts_per_marker_sampled_DPR)
b9_native =  marker_edges(9,pts_per_marker_sampled_DPR)

b7_wrt_obj =  tf_8_7.dot(b7_native)
b8_wrt_obj =  b8_native
b9_wrt_obj =  tf_8_9.dot(b9_native)

all_points_wrt_obj = np.hstack((b7_wrt_obj,b8_wrt_obj,b9_wrt_obj))

with np.load('HD310.npz') as X:
	mtx, dist = [X[i] for i in ('mtx','dist')] 

cap = cv2.VideoCapture(0)

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

emptyList = [0]*3

j = 0  # iteration counter

t0 = time.time() 
time_vect = [0]*iterations_for_while

# Capture frame-by-frame
#frame = cv2.imread('aruco_7_8_9.jpg')
#frame_color = frame
#frame_gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#frame = frame_gray

while(j<iterations_for_while):
	# Capture frame-by-frame
	ret, frame = cap.read()
	frame_color = np.copy(frame)
	frame_gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame_gray_draw = np.copy(frame_gray)

	# cv2.imshow('frame',frame)
	# if cv2.waitKey(1) & 0xFF == ord('q') or j >= iterations_for_while:
	# 	break

	
	# frame = frame_gray
	# analyze with one image
	# frame = cv2.imread('opencv_frame_1.jpg')
	# frame_color = frame
	# frame_gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# frame = frame_gray
	
	#lists of ids and the corners beloning to each id
	corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
	
	if ids is not None:
		if 7 in ids and 8 in ids and 9 in ids:
			ind_7 = np.where(ids == 7)[0][0]
			ind_8 = np.where(ids == 8)[0][0]
			ind_9 = np.where(ids == 9)[0][0]
			# print ind_7, ind_8, ind_9,"ind_7, ind_8, ind_9," 
			stkd_2d_corn_pix_sp[0:4,:] = corners[ind_7][0]
			stkd_2d_corn_pix_sp[4:8,:] = corners[ind_8][0]
			stkd_2d_corn_pix_sp[8:12,:] = corners[ind_9][0]
			
			# local frame normalization
			stacked_corners = np.reshape (stkd_2d_corn_pix_sp, (3,4,2)) # 3 is the number of markers seen now
			frame_gray_norm,frame_grad_u, frame_grad_v, frame_grad_u_img, frame_grad_v_img = normalize_markers (frame_gray,ids,stacked_corners,dilate_fac,thresh_low,thresh_high)

			# print frame_grad_v,"frame_grad_v"
			# print frame_grad_v.shape,"frame_grad_v.shape"

			emptyList[0],emptyList[1],emptyList[2] = corners[ind_7], corners[ind_8], corners[ind_9]
	#            frame = aruco.drawDetectedMarkers(frame, emptyList, np.array([ids[ind_7],ids[ind_8],ids[ind_9]]))
			rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers( emptyList, marker_size_in_mm, mtx,dist)
			pose_marker_without_opt[j,0:3] = rvecs[1][0]
			pose_marker_without_opt[j,3:6] = tvecs[1][0]
			x_guess = np.append(rvecs[1],tvecs[1])
			tf_c_8_ArUco = x_guess
			
			t_1_APE = time.time()
	#        res = minimize(objective_fun_NM,x_guess,method='nelder-mead',options={'xtol': 1e-8, 'disp': True})
			res = leastsq (objective_fun_LM,x_guess,Dfun=None, full_output=0, col_deriv=0, ftol=1.49012e-4, xtol=1.49012e-4, gtol=0.0, maxfev=0, epsfcn=None, factor=100, diag=None) 
			t_2_APE = time.time() -t_1_APE
			
	#            print(t_2_APE,"t_2_APE")
			pose_marker_with_APE[j,:] = res[0]
		
			draw_synth_img(frame,pose_marker_with_APE[j,:],7)
		


			tf_c_8 = RodriguesToTransf(res[0]) # for Leveneberg-Marquardt
	#        tf_c_8 = RodriguesToTransf(res.x) # for Nelder-Mead
			tf_c_8_APE = tf_c_8
			

			tf_c_8_APE_rvec,_ = cv2.Rodrigues(tf_c_8_APE[0:3,0:3])
			tf_c_8_APE_tvec   = tf_c_8_APE[0:3,3]
			tf_c_8_APE_6x1 = np.append(tf_c_8_APE_rvec,tf_c_8_APE_tvec)
			frame_grad_u_col = cv2.cvtColor(frame_grad_u_img,cv2.COLOR_GRAY2BGR)
			frame_grad_v_col = cv2.cvtColor(frame_grad_v_img,cv2.COLOR_GRAY2BGR)
			# draw_marker_edges(frame_grad_v_col,tf_c_8_APE_6x1,8)
			# draw_marker_edges(frame_grad_u_col,tf_c_8_APE_6x1,8)
			pert = np.array([0, 0, 0, 0,1,0])

			
			find_Jac_and_err ( tf_c_8_APE_6x1,frame_grad_u,frame_grad_v )
			find_Jac_and_err ( tf_c_8_APE_6x1+pert,frame_grad_u,frame_grad_v )
			

			# draw_marker_edges(frame_gray_draw,tf_c_8_APE_6x1,8,(0,0,255))
			
			# draw_marker_edges(frame_gray_draw,tf_c_8_APE_6x1+pert,8,(255,255,255))
			
			# draw_marker_edges(frame_color,tf_c_8_APE_6x1,9,(0,0,255))
			# draw_marker_edges(frame_color,tf_c_8_APE_6x1+pert,7,(0,255,0))
			# draw_marker_edges(frame_color,tf_c_8_APE_6x1+pert,8,(0,255,0))
			# draw_marker_edges(frame_color,tf_c_8_APE_6x1+pert,9,(0,255,0))
# def partial_u_wrt_x_hat(mtx,tvecs):
#             t_dpr_1=time.time()
#             pert = np.array([0.08,0.08,0.08,1,1,1])
#             res_GN =  gauss_newton (tf_c_8_APE_6x1) 
	
#     #        res = leastsq (objective_fun_LM_DPR,tf_c_8_APE_6x1,Dfun=None, full_output=0, col_deriv=0, ftol=1.49012e-2, xtol=1.49012e-2, gtol=0.0, maxfev=0, epsfcn=None, factor=100, diag=None) 
#             t_dpr_2 = time.time() - t_dpr_1
			
#             print ("t_dpr_2",t_dpr_2)    
# #            
# #            
# #    #        print ("t_dpr_2",t_dpr_2)    
# #    #            
#             tf_c_8_DPR = RodriguesToTransf(res_GN) 
#             pose_marker_with_DPR[j,:] = res_GN
#    
	#                cv2.circle( frame, tuple(np.ndarray.astype(np.array([0,100]),int)) , 5, (255,255,255), -1)
				
	#    ##drawing
				
#            for l in range(12):
##                cv2.circle(frame, tuple(np.ndarray.astylpe(transf_and_proj[l,:],int)) , 2, (0,0,255), -1)
	##                
	
	####imaging
			print (j)
			j = j+1
			# cv2.imshow('frame',frame)
			# cv2.imshow('frame_color',frame_color)
			cv2.imshow('frame_gray',frame_gray_draw)
			# cv2.imshow('frame_grad_u_img',frame_grad_u_img)
			# cv2.imshow('frame_grad_v_img',frame_grad_v_img)
			# cv2.imshow('frame_gray',frame_gray)

			if cv2.waitKey(0) & 0xFF == ord('q') or j >= iterations_for_while:
				break
	else: 
		print("all markers are not visible")
		cv2.imshow('frame',frame)
		if cv2.waitKey(1) & 0xFF == ord('q') or j >= iterations_for_while:
			break


#cv2.destroyAllWindows()



#### Analysis


pose_marker_with_APE= pose_marker_with_APE[0:j,:]

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
#    print (np.std(tip_posit,axis=0), "std in x,y,z")
	ax.scatter(tip_posit[:,0],tip_posit[:,1],tip_posit[:,2],c=color)
else:
	print ("the end")
#    print (np.std(pose_marker,axis=0), "std in x,y,z,axangle")
#    print (np.std(pose_marker,axis=0,bias =1), "std in x,y,z,axangle")
#    print (np.var(pose_marker,axis=0), "var in x,y,z,axangle")
#    # sens_noise_cov_mat = np.cov(pose_marker_without_opt.T)
	ax.scatter(pose_marker_without_opt[:,3],pose_marker_without_opt[:,4],pose_marker_without_opt[:,5],c ='r')
	ax.scatter(pose_marker_with_APE[:,3],pose_marker_with_APE[:,4],pose_marker_with_APE[:,5],c = 'b' )
	# ax.scatter(pose_marker_with_DPR[:,3],pose_marker_with_DPR[:,4],pose_marker_with_DPR[:,5],c = 'g' )
# np.savetxt("/home/biorobotics/Desktop/tejas/cpp_test/workingCodes/noise_filter/sens_noise_cov_mat.txt",sens_noise_cov_mat)

if hist_plot_switch == 1:
	# fig = plt.figure()

	# plt.hist(pose_marker[:,0],20,facecolor='red',density=True)
	# plt.hist(pose_marker[:,1],20,facecolor='green',density=True)
	# plt.hist(pose_marker[:,2],20,facecolor='blue',density=True)
	# fig = plt.figure()
	# plt.hist(pose_marker[:,0],20,facecolor='red',density=True)
	# fig = plt.figure()
	# plt.hist(pose_marker[:,1],20,facecolor='green',density=True)
	# fig = plt.figure()
	# plt.hist(pose_marker[:,2],20,facecolor='blue',density=True)

#    fig = plt.figure()
#    plt.hist(pose_marker_avg[:,2],20,facecolor='red',density=True, label='pose_marker_avg')
#    plt.legend()
	fig = plt.figure()
	plt.hist(pose_marker_with_APE[:,5],200,facecolor='green',density=True,  label='with_opt')
#    plt.legend()
#    fig = plt.figure()
	plt.hist(pose_marker_without_opt[:,5],200,facecolor='red',density=True,  label='without_opt')
	plt.legend()

	# fig = plt.figure()
	# plt.hist(pose_marker[:,2],20,facecolor='blue',density=True)


# plt.show()

