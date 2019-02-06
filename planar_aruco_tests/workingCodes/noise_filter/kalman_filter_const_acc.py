#Used this code to confirm that the tvec and rvec given by the 
 # estimatePoseSingleMarkers is of the marker frame wrt the camera frame



###

import numpy as np
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
from mpl_toolkits.mplot3d import Axes3D
import transforms3d as tf3d
import time

def RodriguesToTransf(x):
    #input (6,)
    x = np.array(x)
    # print x[0:3]
    
    rot,_ = cv2.Rodrigues(x[0:3])
    
    
    trans =  np.reshape(x[3:6],(3,1))

    Trransf = np.concatenate((rot,trans),axis = 1)
    Trransf = np.concatenate((Trransf,np.array([[0,0,0,1]])),axis = 0)

    return Trransf

def create_A_mat (dt):
    for i in range(12):
        A_empty[i,i+6] = dt
    for i in range(6):
        A_empty[i,i+12] = dt*dt/2      #####constant velocity state model, thetax, thetay thetaz are axis angles.

    return A_empty

def KF(mu_tm1, cov_tm1, z_t):
    z_t = z_t.reshape(6,1)
    random_mat = np.random.random((18,18))
    R_t = (random_mat+random_mat.T)*R_t_coeff
    
    A_t = create_A_mat(dt)
    mu_t_bar = A_t.dot(mu_tm1) 

    # print mu_t_bar, "mu_t_bar"
    cov_t_bar = A_t.dot(cov_tm1).dot(A_t.T) + R_t
    # print cov_t_bar,"cov_t_bar"
    K_t = cov_t_bar.dot(C_t.T).dot(np.linalg.inv(C_t.dot(cov_t_bar).dot(C_t.T) + Q))
    # print (cov_t_bar).dot(C_t.T), "(cov_t_bar).dot(C_t.T)"
    # print K_t
    mu_t = mu_t_bar + K_t.dot(z_t - C_t.dot(mu_t_bar))
    cov_t = (np.eye(np.shape(cov_tm1)[0]) - K_t.dot(C_t)).dot(cov_t_bar)
    print np.linalg.norm(K_t), "cov_t"
    # print mu_t.shape
    # print mu_t[6:12,0]
    # print "..   "
    return mu_t,cov_t



with np.load('B4Aruco.npz') as X:
    mtx, dist = [X[i] for i in ('mtx','dist')] 


### Switches: 
sub_pix_refinement_switch = 1
detect_tip_switch = 0
hist_plot_switch = 0
kalman_filter_switch = 1

    
iterations_for_while = 1000
marker_size_in_mm = 19.16
R_t_coeff = 1000
noise_coeff = 100
state_cov_mat_t0_coeff = 01000     
tip_coord  = np.array([ 2.89509534, -111.83311787  , -2.33105497,1]) 


pose_marker = np.zeros((iterations_for_while,6))
pose_marker_filtered = np.zeros((iterations_for_while,6))
pose_marker_dot = np.zeros((iterations_for_while,6))
pose_marker_dot_dot = np.zeros((iterations_for_while,6))
tip_posit = np.zeros((iterations_for_while,3))
color = [0]*iterations_for_while


aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters =  aruco.DetectorParameters_create()
parameters.cornerRefinementMethod = sub_pix_refinement_switch
parameters.cornerRefinementMinAccuracy = 0.05
emptyList = list()
emptyList.append(np.zeros((4,4)))
time_vect = [0]*iterations_for_while


###### define matrices 
global A_empty, C_t, K_t, dt  ####refer sebastian thrun page 34 for notation
A_empty = np.eye(18)   ###12x12
C_t = np.concatenate((np.eye(6),np.zeros((6,12))),axis=1)
Q = np.loadtxt("sens_noise_cov_mat.txt")
Q = Q*noise_coeff
# random_mat = np.random.random((18,18))
# R_t = (random_mat+random_mat.T)*R_t_coeff
state_cov_mat_tm1 = np.eye(18)*state_cov_mat_t0_coeff
state_tm1 = np.zeros((18,1))
# pred_cov = 


cap = cv2.VideoCapture(0)
t0 = time.time() 
j = 0
while(j<iterations_for_while):
    # Capture frame-by-frame
    ret, frame = cap.read()


 
    #print(parameters)
 
    '''    detectMarkers(...)
        detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
        mgPoints]]]]) -> corners, ids, rejectedImgPoints
        '''
        #lists of ids and the corners beloning to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)


    if ids is not  None :
        for i in range(0,len(ids)):
            if ids[i] ==7:  
                emptyList[0] = corners[i]
                # print corners[i]
                frame = aruco.drawDetectedMarkers(frame, emptyList)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers( emptyList, marker_size_in_mm, mtx,dist)
                transf_mat_for_frame = RodriguesToTransf(np.append(rvecs,tvecs))
                tip_loc_cam_frame = transf_mat_for_frame.dot(tip_coord.T)

        tip_posit[j,:] = tip_loc_cam_frame[0:3].T 
        # rvecs_euler = np.array(tf3d.euler.axangle2euler(rvecs[0,0],np.linalg.norm(rvecs)))
        
        pose_marker[j,:] = np.append(tvecs,rvecs )
        # print pose_marker[j,:], "pose_marker[j,:]"
        color[j] = j 
        time_vect[j] = time.time() - t0
        rot,_ = cv2.Rodrigues(rvecs)
        dt = time_vect[j] - time_vect[j-1]
        
        # print state_tm1.shape
        if j > 0:
            state_t, state_cov_mat_t =KF(state_tm1,state_cov_mat_tm1,pose_marker[j,:].T)
            # print state_t,"state_t"
            # print pose_marker[j,:] , "pose_marker[j,:]"              
            state_tm1 = state_t
            state_cov_mat_tm1 = state_cov_mat_t
        else: 
            # pose_marker_dot[j,:] = (pose_marker[j,:]-pose_marker[j-1,:])/(dt)
            state_tm1[0:6,0] = pose_marker[0,:]
            print state_tm1, "........"
            # state_tm1 = np.append(pose_marker[j,:],pose_marker_dot[j,:])
            state_tm1 = state_tm1.reshape(18,1)

        pose_marker_filtered[j,:] = state_tm1[0:6].reshape(6,)           
        aruco.drawAxis(frame, mtx, dist, rvecs, tvecs , 50)
        aruco.drawAxis(frame, mtx, dist, pose_marker_filtered[j,3:6], pose_marker_filtered[j,0:3] , 20)
        j = j+1
        print j
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or j >= iterations_for_while:
            break

cap.release()
cv2.destroyAllWindows()


pose_marker = pose_marker[0:j,:]
pose_marker_filtered = pose_marker_filtered[0:j,:]
time_vect = time_vect[0:j]

tip_posit = tip_posit[0:j,:]



fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

if detect_tip_switch == 1:
    print np.std(tip_posit,axis=0), "std in x,y,z"
    print np.std(pose_marker_filtered,axis=0), "filtered std in x,y,z"
    ax.scatter(tip_posit[:,0],tip_posit[:,1],tip_posit[:,2],c=color)
else:
    print np.std(pose_marker,axis=0), "std in x,y,z,axangle"
    print pose_marker_filtered.shape
    print np.std(pose_marker_filtered,axis=0), "filtered std in x,y,z"

    fig.suptitle('unfiltered', fontsize=16)
    sens_noise_cov_mat = np.cov(pose_marker.T)
    ax.scatter(pose_marker[:,0],pose_marker[:,1],pose_marker[:,2],c = color[0:pose_marker.shape[0]])
    

    fig = plt.figure()
    fig.suptitle('kalman filtered', fontsize=16)
    ax = fig.add_subplot(111,projection="3d")
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.scatter(pose_marker_filtered[:,0],pose_marker_filtered[:,1],pose_marker_filtered[:,2],c = color[0:pose_marker.shape[0]])
    
    fig = plt.figure()
    fig.suptitle('plotted together', fontsize=16)
    ax = fig.add_subplot(111,projection="3d")
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.scatter(pose_marker_filtered[:,0],pose_marker_filtered[:,1],pose_marker_filtered[:,2])
    ax.scatter(pose_marker[:,0],pose_marker[:,1],pose_marker[:,2])

if hist_plot_switch == 1:
    fig = plt.figure()

    plt.hist(pose_marker[:,0],20,facecolor='red',density=True)
    plt.hist(pose_marker[:,1],20,facecolor='green',density=True)
    plt.hist(pose_marker[:,2],20,facecolor='blue',density=True)

    fig = plt.figure()
    plt.hist(pose_marker[:,0],20,facecolor='red',density=True)
    fig = plt.figure()
    plt.hist(pose_marker[:,1],20,facecolor='green',density=True)
    fig = plt.figure()
    plt.hist(pose_marker[:,2],20,facecolor='blue',density=True)


plt.show()

