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
from scipy.optimize import minimize, leastsq,least_squares
from scipy import linalg
from scipy.spatial import distance
import rospy
from roscam import RosCam


    
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
    
 

def LM_APE_Dodecapen(X, stacked_corners_px_sp, ids):
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
        # print(int(ids[ii]))
        Tf_cent_face,Tf_face_cent = tf_mat_dodeca_pen(int(ids[ii]))
        corners_in_cart_sp[ii,:,:] = Tf_cam_ball.dot(corners_3d(Tf_cent_face,marker_size_in_mm)).T[:,0:3]
    
    corners_in_cart_sp = corners_in_cart_sp.reshape(ids.shape[0]*4,3)
    projected_in_pix_sp,_ = cv2.projectPoints(corners_in_cart_sp,np.zeros((3,1)),np.zeros((3,1)),mtx,dist) 
    projected_in_pix_sp = projected_in_pix_sp.reshape(projected_in_pix_sp.shape[0],2)
    n,_=np.shape(stacked_corners_px_sp)
    V= LA.norm(stacked_corners_px_sp-projected_in_pix_sp, axis=1)
    # print(V,"LM Error")
    return V


def tf_mat_dodeca_pen(face_id):
    '''
    Function that looks at the dodecahedron geometry to get the rotation matrix and translation
    TODO: when in class have to pass the dodecahedron geometry to this as a variable
    Inputs: face_id: the face for which the transformation matrices is quer=ries (int)
    Outputs: T_mat_cent_face = transformation matrix from center of the dodecahedron to a face
    T_mat_face_cent = transformation matrix from face (with given face id) to the dodecahedron center
    
    '''
    T_cent_face_curr = T_cent_face[face_id-1,:,:]
    R_cent_face_curr = R_cent_face[face_id-1,:,:]
    T_mat_cent_face = np.vstack((np.hstack((R_cent_face_curr,T_cent_face_curr)),np.array([0,0,0,1])))
    T_mat_face_cent = np.vstack((np.hstack((R_cent_face_curr.T,-R_cent_face_curr.T.dot(T_cent_face_curr))),np.array([0,0,0,1])))
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

def remove_bad_aruco_centers(center_transforms):

    """
    takes in the tranforms for the aruco centers

    returns the transforms, centers coordinates, and indices for centers which

    aren't too far from the others

    Input: center_transforms  = N transformation matrices stacked as [N,4,4] numpy arrays
    Output = center_transforms[good_indices, :, :] -> accepted center transforms,
    centers_R3[good_indices, :] = center estimates from accepted center transforms,
    good_indices = accepted ids
    """
    max_dist = 50 # mm 

    centers_R3 = center_transforms[:, 0:3, 3]

    distances = distance.cdist(centers_R3, centers_R3)

    good_pairs = (distances > 0) * (distances < max_dist)

    good_indices = np.where(np.sum(good_pairs, axis=0) > 0)[0].flatten()
    # print(good_indices,'good index shape')

    if good_indices.shape[0] == 0 :
        print('good_indices is none, resetting')
        good_indices = np.array([0, 1]) 

    return center_transforms[good_indices, :, :], centers_R3[good_indices, :], good_indices

def local_frame_grads (frame_gray, corners, ids):
''' Takes in the frame, the corners of the markers the camera sees and ids of the markers seen. 
Returns the frame gradients
Input: frame_gray --> grayscale frame
corners: stacked as corners[num_markers,:,:] 
ids: ids seen
Output
frame_grad_u and frame_grad_v: matrices of sizes as frame gray. the areas near the markers will 
have gradients of the frame_gray frame in the same locations and rest is 0
'''
    pass

def marker_edges(aruco_id,downsample):
''' Function to give the edge points in the image and their intensities.
to be called only once in the entire program to gather reference data for DPR
output: 
b_edge = [id,:,:] points on the marker in R3 where the intensities change form [x,y,0]
to be directly used in cv2.projectpoints
edge_intensities = [id,:] ordered expected intensity points for the edge points to be used on obj fun of DPR
'''
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

def LM_DPR(pose_guess, frame, reference_points, reference_intensities):
''' Objective function for the DPR step. Takes in pose as the first arg [mandatory!!] and,
returns the value of the obj fun and jacobial of the obj fun'''    
    pass

    
######


### Switches: 
global transf_and_proj_int 

iii = 0

sub_pix_refinement_switch =1
detect_tip_switch = 0
hist_plot_switch = 0


iterations_for_while =5500
marker_size_in_mm = 17.78
distance_betn_markers = 34.026  #in mm
dilate_fac = 1.1 # dilate the square around the marker
thresh_low = 0
thresh_high = 1
tip_coord  = np.array([3.97135363, -116.99921549 ,-5.32922903,1]) 
gauss_newt_tol = 1e-5
alpha = 0.5
AG_2 = 1e-4




n_points_per_marker = 15


with np.load('PTGREY.npz') as X:
    mtx, dist = [X[i] for i in ('mtx','dist')] 
    
R_cent_face = np.load('Center_face_rotations.npy')
T_cent_face = np.load('Center_face_translations.npy')



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



cv2.namedWindow('image',cv2.WINDOW_NORMAL)
rospy.init_node('RosCam', anonymous=True)
ic = RosCam("/camera/image_color")
while(j<iterations_for_while):
        
    frame = ic.cv_image
    # print(frame)
    if frame is None:
        time.sleep(0.1)
        print("No image")
        continue

    # frame = run()
    # cv2.imshow("image",frame)
    # h,  w = frame.shape[:2]
    # newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    # frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # frame = frame/255
    # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #lists of ids and the corners beloning to each id

    # the first row will allways be [0,0,0] this is to ensure that we can start from face 1 which is actually face 0
    rvecs = np.zeros((13,1,3))
    tvecs = np.zeros((13,1,3))
    # print(ids,"ids")
    markers_possible = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
    markers_impossible = np.array([13,17,37,16,34,45,38,24,47,32,40])
    
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if ids not in markers_impossible and ids is not None and len(ids) >= 2: 
        N_markers =ids.shape[0]
        frame = aruco.drawDetectedMarkers(frame, corners, ids)
        # m is the index for marker ids
        jj = 0
        # the following are with the camera frame
        cent_in_R3 = np.zeros((N_markers,3))
        T_cent = np.zeros((ids.shape[0],4,4)) 
        for m in ids:
            m_indx = np.asarray(np.where(m==ids))
            rvecs[m,:,:], tvecs[m,:,:], _ = cv2.aruco.estimatePoseSingleMarkers( corners[int(m_indx[0])], marker_size_in_mm, mtx,dist)
            T_4_Aruco = RodriguesToTransf(np.append(rvecs[m,:,:], tvecs[m,:,:]))
            T_mat_cent_face,T_mat_face_cent = tf_mat_dodeca_pen(int(m))
            T_cent[jj,:,:] = np.matmul(T_4_Aruco,T_mat_face_cent)
            # cent_in_R3[jj,:] =np.reshape(T_cent[jj,0:3,3],(1,3))
            jj+=1
        # print(T_cent.shape[0],'Markers seen')
        T_cent_accepted, centers_R3, good_indices = remove_bad_aruco_centers(T_cent)

    # this part averages the Trans mats obtained from the ids and finds the mean and SD and kicks outliers

        # sum_pts = np.sum(cent_in_R3,axis=1)
        
        # mean_pts = np.mean(sum_pts,axis = 0)
        # sd_points = np.std(sum_pts,axis=0)
        
        # val_out = [x for x in sum_pts if x > mean_pts - 1.60*sd_points ]
        # val_out = [x for x in val_out if x < mean_pts + 1.60*sd_points ]

        # A = np.int64(val_out)
        # B = np.int64(sum_pts)
        
        # accepted_ids = np.where(A==B)
        # accepted_ids = np.asarray(accepted_ids)
        # print(accepted_ids.shape,"accepted_ids")
        # no_of_accepted_points = accepted_ids.shape[1]



####imaging
        # if accepted_ids.shape[1] > 1 and ids.shape[0]>2:
        #     for kk in range(accepted_ids.shape[1]):
        #         T_cent_accepted[kk,:,:] = T_cent[accepted_ids[0,kk],:,:]
        #         px_sp,_ = cv2.projectPoints(np.reshape(cent_in_R3[accepted_ids[0,kk],:],(3,1)).T, np.zeros((3,1)), np.zeros((3,1)), mtx, dist)
        #         cv2.aruco.drawAxis(frame,mtx,dist,rvecs[ids[kk,:],:,:], tvecs[ids[kk],:,:],20)
        #         temp1 = int(px_sp[0,0,0])
        #         temp2 = int(px_sp[0,0,1])
        #         # cv2.circle(frame,(temp1,temp2), 10 , (0,0,255),2 )
                
        # elif accepted_ids.shape[1]== 0:
        #     print("No markers accepted, Frame Dropped")
        #     continue
        # elif ids.shape[0] <2:
        #     print("Less than 2 markers detected. NOT performing APE, Frame Dropped")
        #     continue
        # else:
        #     print("Less than 2 markers acceptable !!")
        #     # cv2.imshow('frame',frame)

        for i in range(len(ids[good_indices])):

            # print(centers_R3[i,:],'id of accepted marker')
            cent_in_R3 = centers_R3[i,:]


            px_sp,_ = cv2.projectPoints(np.reshape(cent_in_R3, (1,3)),
                                        np.zeros((3,1)), np.zeros((3,1)), mtx, dist)

            temp1 = int(px_sp[0,0,0])

            temp2 = int(px_sp[0,0,1])

            # cv2.circle(frame, (temp1,temp2), 10 , (0,0,0), 10)
            
        # Tf_cam_ball = np.mean(T_cent_accepted,axis=0)
        Tf_cam_ball = np.mean(T_cent_accepted,axis=0)
        # print(T_cent_accepted.shape)
        # Tf_cam_ball = T_cent_accepted[0,:,:]
        # getting the rvecs and t vecs by averaging
        r_vec_aruco,_ = cv2.Rodrigues(Tf_cam_ball[0:3,0:3])
        t_vec_aruco = Tf_cam_ball[0:3,3]
        stacked_corners_px_sp =  np.reshape(np.asarray(corners),(ids.shape[0]*4,2))
        
        X_guess = np.append(r_vec_aruco,np.reshape(t_vec_aruco,(3,1))).reshape(6,1)
        pose_marker_without_opt[j,:] = X_guess.T # not efficient. May have to change
        # print(X_guess,"X_guess")
        t_0_APE = time.time()
        # res = least_squares (LM_APE_Dodecapen,np.reshape(X_guess,(6,)),jac='2-point', bounds=(-np.inf, np.inf), method='lm', ftol=1e-08, xtol=1e-08, gtol=1e-15, x_scale=1.0, loss='linear', f_scale=1.0, diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None, max_nfev=None, verbose=0,args = (stacked_corners_px_sp, ids))
        res = leastsq (LM_APE_Dodecapen,X_guess,Dfun=None, full_output=0, col_deriv=0, ftol=1.49012e-8, xtol=1.49012e-8, gtol=0.0, maxfev=1000, epsfcn=None, factor=100, diag=None,args = (stacked_corners_px_sp, ids)) 
        t_1_APE = time.time() - t_0_APE
        # print(t_1_APE,"Time for APE")
        
        # if res.success is not True:
        #     print("APE FAIL !!!")
        #     time.sleep(1)
        

        pose_marker_with_APE[j,:] = np.reshape(res[0],(1,6))
        
        # print(LA.norm(np.reshape(res[0],(1,6)) - X_guess.T ,2),'Improvement in APE')
        Tf_cam_ball = RodriguesToTransf(res[0])
        
        px_sp,_ = cv2.projectPoints(np.reshape(res[0][3:6],(3,1)).T, np.zeros((3,1)), np.zeros((3,1)), mtx, dist)
        temp1 = int(px_sp[0,0,0])
        temp2 = int(px_sp[0,0,1])
        cv2.circle(frame,(temp1,temp2), 8 , (0,0,255), 3)
        no_of_accepted_points = len(good_indices)              
        if no_of_accepted_points is not 0:
            px_sp,_ = cv2.projectPoints(np.reshape(t_vec_aruco,(3,1)).T, np.zeros((3,1)), np.zeros((3,1)), mtx, dist)
            temp1 = int(px_sp[0,0,0])
            temp2 = int(px_sp[0,0,1])
            # cv2.circle(frame,(temp1,temp2), 10 , (0,255,0), 2)

        N_corners = ids.shape[0]
        
        
        corners_in_cart_sp = np.zeros((ids.shape[0],4,3))
        
        for ii in range(ids.shape[0]):
            Tf_cent_face,Tf_face_cent = tf_mat_dodeca_pen(int(ids[ii]))
            corners_in_cart_sp[ii,:,:] = Tf_cam_ball.dot(corners_3d(Tf_cent_face,marker_size_in_mm)).T[:,0:3]
        
        corners_in_cart_sp = corners_in_cart_sp.reshape(ids.shape[0]*4,3)
        projected_in_pix_sp,_ = cv2.projectPoints(corners_in_cart_sp,np.zeros((3,1)),np.zeros((3,1)),mtx,dist) 
        projected_in_pix_sp = projected_in_pix_sp.reshape(projected_in_pix_sp.shape[0],2)
        
        projected_in_pix_sp_int = np.int16(projected_in_pix_sp)
        
        for iii in range(projected_in_pix_sp_int.shape[0]):
            cv2.circle(frame,(projected_in_pix_sp_int[iii,0],projected_in_pix_sp_int[iii,1]), 5 , (0,255,0),-1)
            cv2.circle(frame,(stacked_corners_px_sp[iii,0],stacked_corners_px_sp[iii,1]), 3 , (255,0,0),-1)
            
        # corners_in_cart_sp = corners_in_cart_sp[0:3,:].T
        # projected_corners_in_px_sp = 
        # print (j,'frame_number')
        j = j+1
        # cv2.imshow('frame',frame)
        cv2.imshow('frame_color',frame)

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

# pose_marker_with_DPR= pose_marker_with_DPR[0:j,:]
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
#    print (np.std(pose_marker,axis=0), "std in x,y,z,axangle")
#    print (np.std(pose_marker,axis=0,bias =1), "std in x,y,z,axangle")
#    print (np.var(pose_marker,axis=0), "var in x,y,z,axangle")
#    # sens_noise_cov_mat = np.cov(pose_marker_without_opt.T)
    ax.scatter(pose_marker_without_opt[:,3],pose_marker_without_opt[:,4],pose_marker_without_opt[:,5],c ='k')
    ax.scatter(pose_marker_with_APE[:,3],pose_marker_with_APE[:,4],pose_marker_with_APE[:,5],c = 'r' )
    # ax.scatter(pose_marker_with_DPR[:,3],pose_marker_with_DPR[:,4],pose_marker_with_DPR[:,5],c = 'g' )
    
#     # ax.set_xlim((np.min(pose_marker_without_opt[:,3]),np.max(pose_marker_without_opt[:,3])))
#     # ax.set_ylim((np.min(pose_marker_without_opt[:,4]),np.max(pose_marker_without_opt[:,4])))
#     # ax.set_zlim((np.min(pose_marker_without_opt[:,5]),np.max(pose_marker_without_opt[:,5])))
    
#     # ax.set_xlim((-100,100))
#     # ax.set_ylim((-100,100))
#     # ax.set_zlim((200,350))
    
    plt.axis('equal')
# np.savetxt("/home/biorobotics/Desktop/tejas/cpp_test/workingCodes/noise_filter/sens_noise_cov_mat.txt",sens_noise_cov_mat)
# print (pose_marker_without_opt[:,2]), "histogram z "
if hist_plot_switch == 1:
    fig = plt.figure()
    
    # plt.hist(pose_marker_without_opt[:,2],j,facecolor='blue',normed = 1 )
    # plt.hist(pose_marker_without_opt[:,2],j,facecolor='blue',normed = 1 )
 
    plt.hist(pose_marker_without_opt[:,2],j,facecolor='blue',normed = 1,label = 'without_opt' )
    fig = plt.figure()
    plt.hist(pose_marker_with_APE[:,2],j,facecolor='red',normed = 1, label = 'with_opt'  )

    # plt.hist(pose_marker[:,0],20,facecolor='red',density=True)
    # fig = plt.figure()
    # plt.hist(pose_marker[:,1],20,facecolor='green',density=True)
    # fig = plt.figure()
    # plt.hist(pose_marker[:,2],20,facecolor='blue',density=True)

#    fig = plt.figure()
#    plt.hist(pose_marker_avg[:,2],20,facecolor='red',density=True, label='pose_marker_avg')
#    plt.legend()
#     fig = plt.figure()
#     plt.hist(pose_marker_with_opt[:,5],200,facecolor='green',density=True,  label='with_opt')
# #    plt.legend()
# #    fig = plt.figure()
#     plt.hist(pose_marker_without_opt[:,5],200,facecolor='red',density=True,  label='without_opt')
#     plt.legend()




plt.show()

