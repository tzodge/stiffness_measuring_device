from __future__ import print_function, division
import numpy as np
from numpy import linalg as LA
import cv2
import cv2.aruco as aruco
import time
from scipy.interpolate import griddata
import argparse
with np.load('HD310.npz') as X:
    mtx, dist = [X[i] for i in ('mtx','dist')] 


 
err_tol = 1e-6
tra_tol = 1e-6
ang_tol = 1e-6
max_iter_GN = 1000
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

 
aruco_id = 8
# print (img_pnts[aruco_id].shape)
# print (max(img_pnts[aruco_id][:,0]))


frame = cv2.imread("synth_img_{}_transl_200.jpg".format(aruco_id))
frame_gray = cv2.imread("synth_img_{}_transl_200.jpg".format(aruco_id),0)
frame_gray_draw = np.copy(frame_gray)
frame_gray_int16 = np.int16(frame_gray_draw)

parameters =  aruco.DetectorParameters_create()
parameters.cornerRefinementMethod = 1
parameters.cornerRefinementMinAccuracy = 0.05
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

corners, ids, rejectedImgPoints = aruco.detectMarkers(frame_gray, aruco_dict, parameters=parameters)
corners =  np.reshape(np.asarray(corners),(ids.shape[0],4,2))
# # here b is a matrix of coordinates with respect to the centre of the marker  
# b = edge_pts_in_img_sp[aruco_id]
# b[:,2] = 0.
# b[:,3] = 1
# n = b.shape[0]
# interval = int(n/approx_points)


# print (n,'n original')
# b = b[0::interval,:]
# img_pnts[aruco_id] =img_pnts[aruco_id][0::interval,:]
# n = b.shape[0]
# print (n,'n shortened')


# b_temp = b[:,0:3].reshape(n,3).astype(np.float32) 


pose_corr = np.array([0.,0.,0.,0.,0.,200.])
# pose = pose_corr + np.array([0.05,0.05,0.05,np.random.rand(),np.random.rand(),np.random.rand()])
pose = pose_corr + np.array([0.00,0.00,0.00,1.,1.,11])
print("pose",pose)

def get_marker_borders (corners,dilate_fac):

    cent = np.array([np.mean(corners[:,0]), np.mean(corners[:,1])])
    
    vert_1 = (corners[0,:] - cent)* dilate_fac
    vert_2 = (corners[1,:] - cent)* dilate_fac
    vert_3 = (corners[2,:] - cent)* dilate_fac
    vert_4 = (corners[3,:] - cent)* dilate_fac
    
    expanded_corners = np.vstack((vert_1+cent,vert_2+cent,vert_3+cent,vert_4+cent))
     
    return expanded_corners

def local_frame_grads (frame_gray, corners, ids):
    dilate_fac =1.2
    # frame_modified = np.zeros((frame_gray.shape[0],frame_gray.shape[1]))
    frame_grad_u = np.zeros((frame_gray.shape[0],frame_gray.shape[1]))
    frame_grad_v = np.zeros((frame_gray.shape[0],frame_gray.shape[1]))

    for i in range(len(ids)):
        # print(corners[i,:,:].shape,"corners[i,:,:]")
        expanded_corners = get_marker_borders(corners[i,:,:],dilate_fac)
        v_low = int(np.min(expanded_corners[:,1]))
        v_high = int(np.max(expanded_corners[:,1])) 
        u_low = int(np.min(expanded_corners[:,0]))
        u_high = int(np.max(expanded_corners[:,0]))

        frame_local = np.copy(frame_gray[v_low:v_high,u_low:u_high]) # not sure if v and u are correct order
        # frame_modified[x_low:x_high,y_low:y_high] = np.copy(frame_local)

        A,B = np.gradient(frame_local.astype('float32'))
        frame_grad_v[v_low:v_high,u_low:u_high] = np.copy(A) 
        frame_grad_u[v_low:v_high,u_low:u_high] = np.copy(B)
    # cv2.imshow("..", frame_grad_v)
    # time.sleep(1)
    return  frame_grad_u, frame_grad_v

def marker_edges (aruco_id, dwn_smpl):
    

    # frame = cv2.imread("synth_img_{}_transl_200.jpg".format(aruco_id))
    # frame_gray = cv2.imread("synth_img_{}_transl_200.jpg".format(aruco_id),0)
    # frame_gray_draw = np.copy(frame_gray)
    # frame_gray_int16 = np.int16(frame_gray_draw)

    # here b is a matrix of coordinates with respect to the centre of the marker  
    edge_pts_in_img_sp = [0]*13
    aruco_images = [0]*13
    aruco_images_int16 = [0]*13
    img_pnts = [0]*13
    for i in range(1,13):
        edge_pts_in_img_sp[i] = np.loadtxt("thick_edge_coord_R3/id_{}.txt".format(i),delimiter=',',dtype=np.float32)
        aruco_images[i]= cv2.imread("aruco_images_mip_maps/res_38_{}.jpg".format(i),0)
        img_pnts[i] = np.loadtxt("thick_edge_coord_pixels/id_{}.txt".format(i),delimiter=',',dtype='int16')
        aruco_images_int16[i] = np.int16(aruco_images[i])
    b = edge_pts_in_img_sp[aruco_id]
    # print(b)
    b[:,2] = 0.
    b[:,3] = 1
    n = b.shape[0]
    interval = int(n/dwn_smpl)
    print(interval,"intr")
    print (n,'n original')
    b = b[0::interval,:]
    img_pnts[aruco_id] =img_pnts[aruco_id][0::interval,:]
    n = b.shape[0]
    print (n,'n shortened')
    print (img_pnts[aruco_id].shape)
    print (max(img_pnts[aruco_id][:,0]))

    b_temp = b[:,0:3].reshape(n,3).astype(np.float32) 
    edge_intensities = aruco_images_int16[aruco_id][img_pnts[aruco_id][:,1]+60,img_pnts[aruco_id][:,0]+60] # can we have it in terms of dil_fac

    return b_temp , edge_intensities

b_temp , edge_intensities_expected = marker_edges (8, 100)

def error_DPR (pose,frame_gray,b_temp,edge_intensities_expected):

    # b_temp , y = marker_edges (8, 100)
    # print(b_temp)
    proj_points, duvec_by_dp_all = cv2.projectPoints( b_temp, pose[0:3], pose[3:6], mtx, dist)
    # print(pose,"pose")
    # proj_points_int = np.ndarray.astype(proj_points,int)
    proj_points_int = np.ndarray.astype(proj_points,int)

    # print(proj_points_int.shape,"proj_points_int.shape")
    # print(proj_points.shape,"proj_points.shape")

    n1 = proj_points_int.shape[0]
    proj_points_int = proj_points_int.reshape(n1,2) ### TODO : reshape takes time
    # proj_points_int = np.unique(proj_points_int,axis=0)
    proj_points = proj_points.reshape(proj_points.shape[0],2)
    ################ EOS 
     
    ### interpolation at pixel values
    ######### interpolation of duvec_by_dp_all  Jacobian components
    du_by_dp = griddata(proj_points,duvec_by_dp_all[0::2,0:6],(proj_points_int[:,0],proj_points_int[:,1]), method = 'nearest')
    dv_by_dp = griddata(proj_points,duvec_by_dp_all[1::2,0:6],(proj_points_int[:,0],proj_points_int[:,1]), method = 'nearest')
    # print(du_by_dp.shape,"du_by_dp.shape")
    
#     du_by_dp_2 = np.zeros((n1,6))        
#     dv_by_dp_2 = np.zeros((n1,6))        

#     du_by_dp_2 = duvec_by_dp_all[0::2,0:6]
#     dv_by_dp_2 = duvec_by_dp_all[0::1,0:6]
#     print (np.allclose(du_by_dp_1,du_by_dp_2))

# ################## debugging
#     du_by_dp = du_by_dp_2 
#     dv_by_dp = dv_by_dp_2
# ################## debugging


    n_int = proj_points_int.shape[0]


    # dI_by_dv,dI_by_du = np.gradient(frame_gray.astype('float32')) # to change this to float 32

    dI_by_dv,dI_by_du = local_frame_grads (frame_gray, corners, ids)
    # cv2.imshow("",grad_frame)

    dI_by_dp = np.zeros((n_int,6))
    for i in range(n_int):
        ui,vi = proj_points_int[i,0], proj_points_int[i,1]
        dI_by_dp[i,:] = dI_by_du [vi,ui] * du_by_dp[i] + dI_by_dv [vi,ui] * dv_by_dp[i]

     
    temp = proj_points.shape[0]
    proj_points = proj_points.reshape(temp,2)
     
    # print np.linalg.inv(dI_by_dp.T.dot(dI_by_dp)), "inv(dI_by_dp.T.dot(dI_by_dp))"
    # y = aruco_images_int16[aruco_id][img_pnts[aruco_id][:,1]+60,img_pnts[aruco_id][:,0]+60]
    f_p = frame_gray_int16[proj_points_int[:,1],proj_points_int[:,0]]
    err = (edge_intensities_expected  - f_p )/float(n_int) # this is the error in the intensities
    ii = 0
    for i in range(n_int):  
        ii+=1       
        center = tuple(np.ndarray.astype(proj_points_int[i,:],int))
        cv2.circle( frame_gray_draw, center , 1 , max(127-ii,0), -1)
        center_2 = tuple(np.ndarray.astype(np.array([img_pnts[aruco_id][i,0]+60,img_pnts[aruco_id][i,1]+60]),int))
        cv2.circle( aruco_images[aruco_id], center_2 , 5 , 127, -1)

    return dI_by_dp, err


def DPR_GN(pose,frame_gray):
    t_start = time.time()
    # for ii in range(max_iter_GN):
    iter_ = 0
    err = np.array([1000,1000,1000,1000,1000,1000])
    dp = np.array([1000,1000,1000,1000,1000,1000])
    print(np.square(dp[0:3]).sum() ,"np.square(err).sum()")
    while np.square(err).sum() <= err_tol and np.square(dp[0:3]).sum() <= ang_tol and np.square(dp[3:6]).sum() <= tra_tol or iter_ <= max_iter_GN:
        iter_ += 1
        # print (iter_)
        # print ("here")
        # This is Jtransp. J del p = -Jtransp. err --> del p = (Jtransp.J)^-1 (-Jtransp. err)
        dI_by_dp, err = error_DPR (pose, frame_gray,b_temp,edge_intensities_expected)

        # dp = np.linalg.inv(dI_by_dp.T.dot(dI_by_dp)).dot(dI_by_dp.T.dot(err))
#       # solving with QR decomposition
        Q,R = np.linalg.qr (dI_by_dp)
        dp =  np.linalg.solve (R,(Q.T.dot(err)))

        pose = pose + dp

        #Armijo_Goldstein Backtracking Line Search
        err_new = np.array([10000000,10000000,10000000,10000000,10000000,10000000])
        # AG BLS parameters 
        AG_2 = 1e-4
        alpha = 0.5
        AG_iter_ = 0
        term_2_of_AG_cond = np.dot(dp,(err.dot(dI_by_dp)))
        # print (err.sum() + AG_2*term_2_of_AG_cond ,"err.sum() + AG_2*term_2_of_AG_cond ")

        # while  err_new.sum() > err.sum() + AG_2*term_2_of_AG_cond and AG_iter_ < 5:
        #   # print(AG_iter_,"AG_iter_")
        #   AG_iter_ += 1
        #   dp = alpha * dp
        #   pose = pose + dp 
        #   dI_by_dp , err_new = error_DPR (pose, frame_gray,b_temp,edge_intensities_expected)
        #   term_2_of_AG_cond = np.dot(dp,(err.dot( dI_by_dp)))




        # if (np.square(err).sum() <= err_tol) and ((np.square(dp[0:3]).sum() <= ang_tol) and (np.square(dp[3:6]).sum() <= tra_tol)):
        #   print (ii,"GN iterations")
        #   print (np.square(err).sum() ,"np.square(err).sum()")
        #   print (np.square(dp[0:3]).sum(), "ang_increment")
        #   print (np.square(dp[3:6]).sum(), "tra_increment")
        #   print ("converged")
        #   break

    t_tot = time.time()-t_start
    print (t_tot,"t_GN" )

    cv2.imshow("frame_gray_draw",frame_gray_draw)
    cv2.imshow("frame_gray",frame_gray)
    cv2.imshow("aruco_images[7]",aruco_images[aruco_id])
    cv2.waitKey(0)
    return pose 


print ("result" ,DPR_GN(pose,frame_gray))

