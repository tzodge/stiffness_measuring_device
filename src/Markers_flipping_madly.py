# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 15:43:27 2019

@author: arkad
"""

#Used this code to confirm that the tvec and rvec given by the 
 # estimatePoseSingleMarkers is of the marker frame wrt the camera frame


from __future__ import print_function, division
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
# from sklearn.preprocessing import normalize
#import numdifftools as nd
# from Pt_grey_roscam import *

import roslib
# roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time

# class image_converter:
	# def __init__(self):
	# 	# self.image_pub = rospy.Publisher("image_topic_2",Image)
	# 	self.image_sub = rospy.Subscriber("/camera/image_color",Image,self.callback)

def get_frame():
	frame = rospy.wait_for_message("/camera/image_color", Image, 5 )
	return frame

def run():
	# ic = image_converter()
	rospy.init_node('image_converter', anonymous=True)
	bridge = CvBridge()
	frame = get_frame()
	img = bridge.imgmsg_to_cv2(frame, "bgr8")
	return img

### functions ##
###Drawing functions 
def draw_marker_edges(frame,pose_obj,id_no,color=(0,255,0)):
    ## for now, the ids are assigned sloppily, later add a dictionary containing marker transformations
    if id_no == 7:
        b = b7_wrt_obj
    elif id_no == 8:
        b = b8_wrt_obj
    elif id_no == 9:
        b = b9_wrt_obj
    print (b[0:3,:].shape,"b[0:3,:].shape")
    points_to_project,_ = cv2.projectPoints(b[0:3,:].T,pose_obj[0:3],pose_obj[3:6],mtx,dist)
    points_to_project = points_to_project.reshape(points_to_project.shape[0],2)
#a = np.fliplr(a.T)   
    points_to_project = np.fliplr(points_to_project.T)
    
    for i in range(points_to_project.shape[1]):
        cv2.circle( frame, tuple(np.ndarray.astype(points_to_project[:,i],int)) , 1 , color, -1)
    cv2.imshow("image_draw",frame)
    cv2.waitKey(0)
    
    



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
    
    b7 = marker_edges(7,40)
    b8 = marker_edges(8,40)
    b9 = marker_edges(9,40)
    
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
    cv2.imshow("frame_now" ,frame)
        
    V = Ic_p
    print(V,"V")
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

def tf_mat_dodeca_pen(face_id):
    # looking at the global variables to get the rotation matrix and translation
    T_cent_face_curr = T_cent_face[face_id-1,:,:]
    R_cent_face_curr = R_cent_face[face_id-1,:,:]
    T_mat_cent_face = np.vstack((np.hstack((R_cent_face_curr,T_cent_face_curr)),np.array([0,0,0,1])))
    T_mat_face_cent = np.vstack((np.hstack((R_cent_face_curr.T,-R_cent_face_curr.T.dot(T_cent_face_curr))),np.array([0,0,0,1])))
    return T_mat_cent_face,T_mat_face_cent

def corners_3d(tf_mat,m_s):
    corn_1 = np.array([-m_s/2.0,  m_s/2.0, 0, 1])
    corn_2 = np.array([ m_s/2.0,  m_s/2.0, 0, 1])
    corn_3 = np.array([ m_s/2.0, -m_s/2.0, 0, 1])
    corn_4 = np.array([-m_s/2.0, -m_s/2.0, 0, 1])
    corn_mf = np.vstack((corn_1,corn_2,corn_3,corn_4))
    corn_pgn_f = tf_mat.dot(corn_mf.T)
    return corn_pgn_f

def marker_edges (marker_id, dwn_smpl,dil_fac):
    
    marker_id_str = "aruco_images" + "\{}.jpg".format(marker_id)  
    gray = cv2.imread(marker_id_str,0)
    
    size_dil =int( dil_fac*gray.shape[0] ) # as the image is square
    print(size_dil,"size_dil")
    new_img = np.ones((size_dil,size_dil))*255 # to make it white
    
    border = int((size_dil - gray.shape[0])/2)  # 60 pixels padding to the edges of the old image
    
    new_img[(border):(-border),(border):(-border)] = gray
    
    small_image = cv2.pyrDown(new_img)     #600-> 300
    small_image = cv2.pyrDown(small_image) #300->150
    small_image = cv2.pyrDown(small_image) # 150->75
    small_image = cv2.pyrDown(small_image) # 75->38
    # small_image = cv2.pyrDown(small_image) # 75->38

    small_image_int8 = np.uint8(small_image) # converting to 8bot int or else canny will not work

    a = cv2.Canny(small_image_int8,100,200,apertureSize = 5)
    a = np.fliplr(a.T) #do not change this!!! #            
    
    scale = a.shape[0]/(marker_size_in_mm*dil_fac)
    a_shift = a.shape[1]/(2*scale)
    edge_points = np.asarray(np.nonzero(a))
    # print(edge_points.shape)
    edge_intensities = small_image[edge_points[0,:],edge_points[1,:]]
    b = np.asarray(edge_points)/scale - a_shift
    print(b.shape,"b")
    z = np.zeros((1,b.shape[1]))
    tr_dash = np.ones((1,b.shape[1]))
    b = np.vstack ((b,z,tr_dash))
    interval =  int(b.shape[1]/dwn_smpl)
    print (interval,"interval " )
    b = b[:,0::interval]
    edge_intensities = edge_intensities[0::interval]
    print(edge_intensities,"edge_intensities")
    return b , edge_intensities

def get_marker_borders (corners,dilate_fac):

    cent = np.array([np.mean(corners[:,0]), np.mean(corners[:,1])])
    
    vert_1 = (corners[0,:] - cent)* dilate_fac
    vert_2 = (corners[1,:] - cent)* dilate_fac
    vert_3 = (corners[2,:] - cent)* dilate_fac
    vert_4 = (corners[3,:] - cent)* dilate_fac
    
    expanded_corners = np.vstack((vert_1+cent,vert_2+cent,vert_3+cent,vert_4+cent))
     
    return expanded_corners

def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 

def normalize_markers (frame,ids,stacked_corners,dilate_fac,thresh_low,thresh_high):
    # this will normalize around the markers, threshold it and will also return a 
    #different frame with the gradients of the frame near the markers
    n_markers = ids.shape[0]
    frame_grad_u = np.copy(frame)
    frame_grad_v = np.copy(frame)
    # initialize required frames
    for i in range(n_markers):
        expanded_corners = get_marker_borders(stacked_corners[i,:,:],dilate_fac)
        
        ### do not change order of indices
        x_low = int(np.min(expanded_corners[:,1]))
        x_high = int(np.max(expanded_corners[:,1])) 
        y_low = int(np.min(expanded_corners[:,0]))
        y_high = int(np.max(expanded_corners[:,0]))
       ### do not change order of indices
       
        frame_norm = frame[x_low:x_high,y_low:y_high]
        # frame_norm = cv2.normalize(frame_norm, dist, thresh_low, thresh_high, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2.imshow("frame_norm",frame_norm)
        frame_norm = scale(frame_norm, thresh_low,thresh_high)
        
#        frame_norm = frame_norm*255
        # _,frame_norm_thresh = cv2.threshold(frame_norm, 127, 255, cv2.THRESH_BINARY)
#        print(frame_norm,"frame_norm")
        # plt.spy(frame_norm)
        # cv2.imshow("frame_norm",frame_norm)
        frame[x_low:x_high,y_low:y_high] = frame_norm
        
        # A = cv2.Sobel(frame[x_low:x_high,y_low:y_high],cv2.CV_64F,1,0,ksize=5)
        # B = cv2.Sobel(frame[x_low:x_high,y_low:y_high],cv2.CV_64F,0,1,ksize=5)
        
        A,B = np.gradient (frame[x_low:x_high,y_low:y_high])
        cv2.imshow("gradient_u",A)
        frame_grad_u[x_low:x_high,y_low:y_high] = B
        frame_grad_v[x_low:x_high,y_low:y_high] = A
        
        # print(A.shape,"A")
        # print("...")
        # print(B.shape,"B")
        # print("...")
        
    return frame, frame_grad_u, frame_grad_v

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
    del_u_del_x_hat = np.array([[fx/tvecs[2], 0, -fx*tvecs[0]/tvecs[2]**2],[0, fy/tvecs[2], -fy*tvecs[1]/tvecs[2]**2 ] ]) 
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
    Q,R = LA.qr(Jac)
    # J_ps_inv = LA.inv(R).dot(Q.T)          
#    cond_no =  LA.cond(J_ps_inv)
    # print(Q.shape,"q.shape")
    # print(R.shape,"r.shape")
    # lambda_reg = 0.5
    # J_sq = Jac.T.dot(Jac)
    # Jac_reg = J_sq + np.diag(J_sq)**2*np.eye(6)

    return Jac,Q,R
 
     
   
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
    O_t = np.reshape(np.hstack((b7_expected_intensities,b8_expected_intensities,b9_expected_intensities)),(Ic_p.shape[0],1))
    O_t = scale(O_t,thresh_low,thresh_high)
    # n = Ic_p.shape[0]
    # print (O_t.shape,"O_t.shape")
    # print (Ic_p.shape,"Ic_p.shape")
    err = O_t - Ic_p
    # print(err,"err")
    for i in range(transf_and_proj.shape[0]):
        cv2.circle( frame_color, tuple(np.ndarray.astype(transf_and_proj[i,:],int)) , 1 , (0,30*iii,0), -1)

    iii += 1
    
    return LA.norm(err,1), err, transf_and_proj_int

#def find_Jac (frame_gray,pose):
#    ### takes image as input and returns 6xN jac_ps_inv
    


# def gauss_newton(p):
    
#     del_p_tol = 1e-8
#     del_Err_tol = 1e-8
#     ii = 0  
    
#     p_GN = np.zeros((1000,6))
#     p_GN [0,:] = p
# #    print (p,"p inside GN")
#     Err_GN = np.zeros((1000,1))
#     del_Err_GN = np.zeros((1000,1))
#     del_p_mag = np.zeros((1000,1))
#     del_p_mag[0,0] = 1e+6
#     del_Err_GN[0,0] = 1e+6
      
#     while (del_p_mag[ii,0] > del_p_tol) or (del_Err_GN[ii,0] > del_Err_tol ):
        
#         #### del_p = Jac_ps_inv*(I_t - I_c)
#         #### we need Jac_ps_inv 
#         #### for jac_ps_inv, we need transf_and_proj_int 
#         Err_GN[ii,0], It_m_Ic, transf_and_proj_int = error_DPR(p_GN[ii,:])
#         _, J_ps_inv = Intensity_Jacobian (p_GN[ii,0:3],p_GN[ii,3:6],mtx,frame_grad_u, frame_grad_v, transf_and_proj_int)
        
# #        J_ps_inv = find_Jac_ps_inv (frame_gray,pose)
        
        
#         p_GN[ii+1,:] = p_GN[ii,:] - J_ps_inv.dot(It_m_Ic)
#         Err_GN[ii+1,0], _, _ = error_DPR(p_GN[ii+1,:])
        
#         del_Err_GN[ii+1,0] = Err_GN[ii+1,0] - Err_GN[ii,0]                  
#         del_p_mag[ii+1,0] = LA.norm((p_GN[ii+1,:] - p_GN[ii,:]),1)
        
# #        print (p_GN[ii,:],"p_GN[{},0]".format(ii))
# #        print (p_GN[ii+1,:],"p_GN[{},0]".format(ii+1))
#         print(del_Err_GN[ii,0],"del_Err_GN[ii,0]")
#         print (del_p_mag[ii,0],"del_p_mag[ii,0]")
#         print (ii, "ii, inside GN")
#         draw_marker_edges(frame_color,p_GN[ii,:],8)
#         ii += 1
        
    
#     p_GN = p_GN [0:ii,:]

#     return p_GN[-1,:]
    
def gauss_newton(p):

    # print(p,'p in gauss newton')
    change_in_p = 1e+6
    iter_1 = 0
    E_new_sc = np.inf
    del_p = np.zeros((6,))
    change_in_obj_fun = 1000
    while change_in_obj_fun > gauss_newt_tol and iter_1<200:
        iter_1 += 1
        E_curr_sc,E_curr_vec,transf_and_proj_int = error_DPR(p+del_p)
        
        
#         _, Q,R = Intensity_Jacobian (p[0:3],p[3:6],mtx,frame_grad_u, frame_grad_v, transf_and_proj_int)
#         # del_p = J_ps_inv.dot(E_curr_vec)
#         # _, Q,R = Intensity_Jacobian (p[0:3],p[3:6],mtx,frame_grad_u, frame_grad_v, transf_and_proj_int)
# #        print(Q.shape)
# #        print(R.shape)
#         c = Q.T.dot(E_curr_vec)
        # print(iter_1,"iter_1")
        Jac,Q,R = Intensity_Jacobian (p[0:3],p[3:6],mtx,frame_grad_u, frame_grad_v, transf_and_proj_int)
        # c = Q.T.dot(-E_curr_vec)
        # del_p = LA.solve(R,c)
        # del_p = np.reshape(del_p,(6,))
        E_curr_vec = np.reshape(E_curr_vec,(E_curr_vec.shape[0],1))
        point_wise_error = np.multiply(E_curr_vec,Jac)
        grad_Err_curr_sc_wrt_p = np.sum(point_wise_error,0)
        
        lambda_reg = 0.5
        J_sq = point_wise_error.T.dot(point_wise_error)
        Jac_reg = J_sq + lambda_reg*np.eye(6)  # np.diag(J_sq)
        
        del_p = linalg.solve_triangular(Jac_reg,Jac.T.dot(E_curr_vec))
        del_p = np.reshape(del_p,(6,))
        
        # print(np.dot(del_p,grad_Err_curr_sc_wrt_p),"np.dot(del_p,grad_Ic_wrt_p)")
        # print(LA.cond(R),"condition_of_R")
        
        # print(del_p,"del_p")
        # draw_marker_edges(frame_color,p,8)
        p = p + del_p
        # print(p,"p_old")
        # E_new_sc,_,_ =  error_DPR(p)
        # print (E_new_sc,"E_new_sc" )
        # print (E_curr_sc,"E_curr_sc")
        j = 1
        E_new_sc = np.inf
        # abc= E_curr_sc + c*np.dot(del_p,grad_Ic_wrt_p)
        # print(abc," E_curr_sc + c*np.dot(del_p,grad_Ic_wrt_p")
        # print(c.shape,"c.shape")
        
        while (E_new_sc > E_curr_sc + AG_2*np.dot(del_p,grad_Err_curr_sc_wrt_p)) and j<35:
            del_p = alpha*del_p
            p_str = p + del_p
            E_new_sc,_,_ = error_DPR(p_str)            
#            print (j,"j") 
            j = j+1
           
        change_in_p = LA.norm(p-p_str,1) 
        change_in_obj_fun = np.abs(E_new_sc - E_curr_sc)
        # print(change_in_p,"error")
        print(change_in_obj_fun,"E_curr_sc")

    return p   

######


### Switches: 
global transf_and_proj_int 

iii = 0

sub_pix_refinement_switch =1
detect_tip_switch = 0
hist_plot_switch = 0


iterations_for_while =5550
marker_size_in_mm = 17.78
distance_betn_markers = 34.026  #in mm
dilate_fac = 1.1 # dilate the square around the marker
thresh_low = 0
thresh_high = 1
tip_coord  = np.array([3.97135363, -116.99921549 ,-5.32922903,1]) 
gauss_newt_tol = 1e-5
alpha = 0.5
AG_2 = 1e-4



tf_8_7, tf_8_9 = tf_mat_pla_pen (distance_betn_markers)
n_points_per_marker = 15
# b7_native, b7_expected_intensities  =  marker_edges(7,n_points_per_marker,dilate_fac)
# b8_native, b8_expected_intensities =  marker_edges(8,n_points_per_marker,dilate_fac)
# b9_native, b9_expected_intensities =  marker_edges(9,n_points_per_marker,dilate_fac)

# b7_wrt_obj=  tf_8_7.dot(b7_native)
# b8_wrt_obj =  b8_native
# b9_wrt_obj =  tf_8_9.dot(b9_native)


with np.load('PTGREY.npz') as X:
    mtx, dist = [X[i] for i in ('mtx','dist')] 
    
R_cent_face = np.load('Center_face_rotations.npy')
T_cent_face = np.load('Center_face_translations.npy')

# cap = cv2.VideoCapture(0)
# ret = cap.set(3,1000)
# ret = cap.set(4,1000)

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
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
while(j<iterations_for_while):
    
    # # Capture frame-by-frame
    
    # ret, frame = cap.read()
    # frame_color = frame

    frame = run()
    cv2.imshow("image",frame)

    # frame = frame/255
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = frame_gray
    
    # # analyze with one image
    
    # frame = cv2.imread('opencv_frame_2.jpg')
    
    # frame = cv2.imread('synthetic_img.jpg')cd 
    # # dist = np.zeros((1,5))  # only to be used for the synthetic image
    # # mtx = np.eye(3)     # only to be used for the synthetic image
    
    # frame_color = frame    
    # frame_gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = frame_gray
    
    #lists of ids and the corners beloning to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    current_marker_1 = 4
    current_marker_2 = 8
#     # the first row will allways be [0,0,0] this is to ensure that we can start from face 1 which is actually face 0
    rvecs = np.zeros((13,1,3))
    tvecs = np.zeros((13,1,3))
    print(ids,"ids")
    markers_possible = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
    markers_impossible = np.array([15,13,17,37,16,34,45,38,24])
    if ids not in markers_impossible and ids is not None:
        N_markers =ids.shape[0]
        frame = aruco.drawDetectedMarkers(frame, corners, ids)
        # m is the index for marker ids
        jj = 0
        for m in ids:
            m_indx = np.asarray(np.where(m==ids))
            rvecs[m,:,:], tvecs[m,:,:], _ = cv2.aruco.estimatePoseSingleMarkers( corners[int(m_indx[0])], marker_size_in_mm, mtx,dist)
            jj+=1
            T_4_Aruco = RodriguesToTransf(np.append(rvecs[m,:,:], tvecs[m,:,:]))
            T_mat_cent_face,T_mat_face_cent = tf_mat_dodeca_pen(int(m))
            cv2.aruco.drawAxis(frame,mtx,dist,rvecs[m,:,:], tvecs[m,:,:],20)
            T_cent = np.matmul(T_4_Aruco,T_mat_face_cent)
            cent_in_R3 =np.reshape(T_cent[0:3,3],(3,1))
            px_sp,_ = cv2.projectPoints(cent_in_R3.T, np.zeros((3,1)), np.zeros((3,1)), mtx, dist)
            temp1 = int(px_sp[0,0,0])
            temp2 = int(px_sp[0,0,1])
            cv2.circle(frame, (temp1,temp2), 10 , (0,0,0), 10)
            cent_prev = cent_in_R3
            print(T_4_Aruco[0:3,0:3])
            print('                             ')
            print(LA.det(T_4_Aruco[0:3,0:3]),"determinant")

# ####imaging
        print (j)
        j = j+1
        cv2.imshow('frame',frame)
        cv2.imshow('frame_color',frame)

        if cv2.waitKey(0) & 0xFF == ord('q') or j >= iterations_for_while:
            break
    else: 
        print("Required marker not visible")
        cv2.imshow('frame',frame)
        if cv2.waitKey(0) & 0xFF == ord('q') or j >= iterations_for_while:
            break


    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break



# #### Analysis


# pose_marker_with_APE= pose_marker_with_APE[0:j,:]

# pose_marker_with_DPR= pose_marker_with_DPR[0:j,:]
# pose_marker_without_opt = pose_marker_without_opt[0:j,:]
# #pose_marker_avg = pose_marker_avg[0:j,:]
# tip_posit = tip_posit[0:j,:]



# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# if detect_tip_switch == 1:
# #    print (np.std(tip_posit,axis=0), "std in x,y,z")
#     ax.scatter(tip_posit[:,0],tip_posit[:,1],tip_posit[:,2],c=color)
# else:
#     print ("the end")
# #    print (np.std(pose_marker,axis=0), "std in x,y,z,axangle")
# #    print (np.std(pose_marker,axis=0,bias =1), "std in x,y,z,axangle")
# #    print (np.var(pose_marker,axis=0), "var in x,y,z,axangle")
# #    # sens_noise_cov_mat = np.cov(pose_marker_without_opt.T)
#     ax.scatter(pose_marker_without_opt[:,3],pose_marker_without_opt[:,4],pose_marker_without_opt[:,5],c ='r')
#     # ax.scatter(pose_marker_with_APE[:,3],pose_marker_with_APE[:,4],pose_marker_with_APE[:,5],c = 'b' )
#     ax.scatter(pose_marker_with_DPR[:,3],pose_marker_with_DPR[:,4],pose_marker_with_DPR[:,5],c = 'g' )
# #    plt.axis('equal')
# # np.savetxt("/home/biorobotics/Desktop/tejas/cpp_test/workingCodes/noise_filter/sens_noise_cov_mat.txt",sens_noise_cov_mat)

# if hist_plot_switch == 1:
#     # fig = plt.figure()

#     # plt.hist(pose_marker[:,0],20,facecolor='red',density=True)
#     # plt.hist(pose_marker[:,1],20,facecolor='green',density=True)
#     # plt.hist(pose_marker[:,2],20,facecolor='blue',density=True)
#     # fig = plt.figure()
#     # plt.hist(pose_marker[:,0],20,facecolor='red',density=True)
#     # fig = plt.figure()
#     # plt.hist(pose_marker[:,1],20,facecolor='green',density=True)
#     # fig = plt.figure()
#     # plt.hist(pose_marker[:,2],20,facecolor='blue',density=True)

# #    fig = plt.figure()
# #    plt.hist(pose_marker_avg[:,2],20,facecolor='red',density=True, label='pose_marker_avg')
# #    plt.legend()
#     fig = plt.figure()
#     plt.hist(pose_marker_with_opt[:,5],200,facecolor='green',density=True,  label='with_opt')
# #    plt.legend()
# #    fig = plt.figure()
#     plt.hist(pose_marker_without_opt[:,5],200,facecolor='red',density=True,  label='without_opt')
#     plt.legend()

#     # fig = plt.figure()
#     # plt.hist(pose_marker[:,2],20,facecolor='blue',density=True)


# #plt.show()

