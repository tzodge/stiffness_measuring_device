import numpy as np 
from numpy import linalg as LA
import cv2

marker_size_in_mm = 17.78
distance_betn_markers = 34.026
gauss_newt_tol = 1e-8
alpha = 0.5
c = 1e-4

### transformation matrix T^i_j from jth frame to ith frame is written here as Ti_j 
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
    
    marker_id_str = "aruco_images" + "\{}.jpg".format(marker_id)  
    gray = cv2.imread(marker_id_str)
    a = cv2.Canny(gray,100,200,apertureSize = 5)
    a[0,:] = 255
    a[-1,:] = 255
    a[:,0] = 255
    a[:,-1] = 255
#    cv2.imshow("image",a)
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
#    corners = np.floor (corners)
#    expanded_corners = np.zeros((4,2))
#    expanded_corners[0,:] = np.array([corners[0,0]-marker_margin, corners[0,1]+ marker_margin])
#    expanded_corners[1,:] = np.array([corners[1,0]+marker_margin, corners[1,1]+ marker_margin])
#    expanded_corners[2,:] = np.array([corners[2,0]+marker_margin, corners[2,1]- marker_margin])
#    expanded_corners[3,:] = np.array([corners[2,0]-marker_margin, corners[3,1]- marker_margin])
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
    frame_grad_u = np.copy(frame)
    frame_grad_v = np.copy(frame)
    # initialize required frames
    for i in range(n_markers):
        expanded_corners = get_marker_borders(stacked_corners[i,:,:],dilate_fac)
        
        ### do not change
        x_low = int(np.min(expanded_corners[:,1]))
        x_high = int(np.max(expanded_corners[:,1])) 
        y_low = int(np.min(expanded_corners[:,0]))
        y_high = int(np.max(expanded_corners[:,0]))
       ### do not change 
        frame_norm = frame[x_low:x_high,y_low:y_high]
        frame_norm = cv2.normalize(frame_norm, frame_norm, thresh_low, thresh_high, cv2.NORM_INF)
        _,frame_norm_thresh = cv2.threshold(frame_norm, 127, 255, cv2.THRESH_BINARY)
        frame[x_low:x_high,y_low:y_high] = frame_norm_thresh
        A,B = np.gradient (frame[x_low:x_high,y_low:y_high])
        frame_grad_u[x_low:x_high,y_low:y_high] = A
        frame_grad_v[x_low:x_high,y_low:y_high] = B
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
    
    
def partial_R_wrt_r (R,r):
    del_R_by_del_r_x = np.zeros((3,3))
    del_R_by_del_r_y = np.zeros((3,3))
    del_R_by_del_r_z = np.zeros((3,3))
    
    norm_r = LA.norm(r,2)
    skw_r = skew_matrix(r)
    
    t_x = np.matmul(np.eye(3)-R, np.array([1,0,0]))
    t_y = np.matmul(np.eye(3)-R, np.array([0,1,0]))
    t_z = np.matmul(np.eye(3)-R, np.array([0,0,1]))
    
    del_R_by_del_r_x = (r[0]*skw_r + skew_matrix(np.cross(r,t_x)))/norm_r**2 .dot(R)
    del_R_by_del_r_y = (r[1]*skw_r + skew_matrix(np.cross(r,t_y)))/norm_r**2 .dot(R)
    del_R_by_del_r_z = (r[2]*skw_r + skew_matrix(np.cross(r,t_z)))/norm_r**2 .dot(R)
    
    del_R_del_r = np.vstack((del_R_by_del_r_x,del_R_by_del_r_y,del_R_by_del_r_z))
    
    return del_R_del_r

def partial_x_hat_wrt_R_hat (tvecs):
    del_tvecs_hat_del_R_hat = np.array([[tvecs[0],tvecs[1],tvecs[2],0,0,0,0,0,0],[0,0,0,tvecs[0],tvecs[1],tvecs[2],0,0,0],[0,0,0,0,0,0,tvecs[0],tvecs[1],tvecs[2]]])
    return del_tvecs_hat_del_R_hat

def partial_u_wrt_x_hat(mtx,tvecs):
    fx = mtx[0,0]
    fy = mtx[1,1]
    del_u_del_x_hat = np.array([fx/tvecs[2], 0, -fx*tvecs[0]/tvecs[2]**2],[0, fy/tvecs[2], -fy*tvecs[1]/tvecs[2]**2 ] ) 
    return del_u_del_x_hat 
        
def partial_Ic_wrt_pix(frame_grad_u, frame_grad_v,u,v):
    del_Ic_del_u  = np.array([0,0])
    
    del_Ic_del_u [0] = frame_grad_u[u,v]
    del_Ic_del_u [1] = frame_grad_v[u,v]
    
    return del_Ic_del_u 

def Intensity_Jacobian (R,rvecs,tvecs,mtx,frame_grad_u, frame_grad_v,transf_and_proj_int):
    
    N = transf_and_proj_int.shape[0]
    Jac = np.zeros(N,6)
    
    for i in range(N):
        
        u,v = transf_and_proj_int[i,0],transf_and_proj_int[i,1]
        del_R_del_r = partial_R_wrt_r (R,rvecs)
        del_x_hat_del_R_hat = partial_x_hat_wrt_R_hat (tvecs)
        del_u_del_x_hat  = partial_u_wrt_x_hat(mtx,tvecs)
        del_Ic_del_u  = partial_Ic_wrt_pix(frame_grad_u, frame_grad_v,u,v)
        
        term_1 = del_u_del_x_hat.dot(del_x_hat_del_R_hat).dot(del_R_del_r)
        
        term_2 = np.hstack((term_1,del_u_del_x_hat))
        
        Jac[i,:] = (del_Ic_del_u.dot(term_2)).reshape(6,)
        
    J_ps_inv = np.linalg.solve(np.matmul(Jac.T,Jac), np.eye(6)).dot(Jac.T)
    return Jac, J_ps_inv
        
   
def error_DPR(p):    
    tf_c_8 = RodriguesToTransf(p)
    
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
    
    return V.sum()/float(n), V

def gauss_newton(p):
    del_p = 1e+5
    while del_p > gauss_newt_tol:
        E_curr_sc,E_curr_vec = error_DPR(p)
        _, J_ps_inv = Intensity_Jacobian (R,rvecs,tvecs,mtx,frame_grad_u, frame_grad_v,transf_and_proj_int)
        del_p = J_ps_inv.dot(E_curr_vec)
        E_new_sc,_ =  error_DPR(p + del_p)
        while (E_new > E_curr_sc):
            del_p = alpha*del_p
            p = p + del_p
            E_new_sc,_ = error_DPR(p)
    

    return p            
        
        
    
    
    
    
    
    