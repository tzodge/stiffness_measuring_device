#Used this code to confirm that the tvec and rvec given by the 
 # estimatePoseSingleMarkers is of the marker frame wrt the camera frame


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



### Switches: 
sub_pix_refinement_switch =1
detect_tip_switch = 0
hist_plot_switch = 0


iterations_for_while = 1000
marker_size_in_mm = 19.16
tip_coord  = np.array([    3.97135363, -116.99921549 ,-5.32922903,1]) 



with np.load('B4Aruco.npz') as X:
    # mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
    mtx, dist = [X[i] for i in ('mtx','dist')] 


print mtx

cap = cv2.VideoCapture(0)

pose_marker = np.zeros((iterations_for_while,6))
tip_posit = np.zeros((iterations_for_while,3))
color = [0]*iterations_for_while


aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters =  aruco.DetectorParameters_create()
parameters.cornerRefinementMethod = sub_pix_refinement_switch
parameters.cornerRefinementMinAccuracy = 0.05


emptyList = list()
emptyList.append(np.zeros((4,4)))


j = 0

t0 = time.time() 
time_vect = [0]*iterations_for_while

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

        pose_marker[j,:] = np.append(tvecs,rvecs*180/np.pi)
        tip_posit[j,:] = tip_loc_cam_frame[0:3].T 
            

        color[j] = j 
        time_vect[j] = time.time() - t0
        rot,_ = cv2.Rodrigues(rvecs)

        j = j+1
        print rvecs.shape, "rvecs" 
        print ".."
        print tvecs.shape, "tvecs"
        print ".."
        print tip_loc_cam_frame, "tip_loc_cam_frame"
        print "..."
        cv2.aruco.drawAxis(frame,mtx,dist,rvecs,tvecs,20)
        # print tvecs.shape, "tvecs.shape"
        drawing_vect = np.zeros((3,1,3))
        drawing_vect[1,0,0] = tip_coord[0]
        drawing_vect[1,0,1] = tip_coord[1]
        drawing_vect[1,0,2] = tip_coord[2]


        pen_tip_2d_coord, _ = cv2.projectPoints(drawing_vect,rvecs,tvecs,mtx,dist)
        print pen_tip_2d_coord,"pen_tip_2d_coord[0]" 
        cv2.line(frame,tuple(np.int32(pen_tip_2d_coord[1,0])),tuple(np.int32(pen_tip_2d_coord[0,0])),(255,255,255),5 )
        cv2.line(frame,tuple(np.int32(pen_tip_2d_coord[2,0])),tuple(np.int32(pen_tip_2d_coord[0,0])),(255,255,255),5 )
        


        drawing_vect = np.zeros((2,1,3))
        drawing_vect[1,0,0] = tip_loc_cam_frame[0]
        drawing_vect[1,0,1] = tip_loc_cam_frame[1]
        drawing_vect[1,0,2] = tip_loc_cam_frame[2]
        rvec_cam = np.zeros((1,1,3))
        tvec_cam = np.zeros((1,1,3))
        pen_tip_optimized, _ = cv2.projectPoints(drawing_vect,rvec_cam,tvec_cam,mtx,dist)
        cv2.line(frame,tuple(np.int32(pen_tip_optimized[1,0])),tuple(np.int32(pen_tip_optimized[0,0])),(0,0,0),5 )
        
        # cv2.line(frame,(0,0),(200,200),(0,0,255),5 )
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or j >= iterations_for_while:
            break

cap.release()
cv2.destroyAllWindows()


pose_marker = pose_marker[0:j,:]
tip_posit = tip_posit[0:j,:]



fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

if detect_tip_switch == 1:
    print np.std(tip_posit,axis=0), "std in x,y,z"
    ax.scatter(tip_posit[:,0],tip_posit[:,1],tip_posit[:,2],c=color)
else:
    print np.std(pose_marker,axis=0), "std in x,y,z,axangle"
    # print np.std(pose_marker,axis=0,bias =1), "std in x,y,z,axangle"
    # print np.var(pose_marker,axis=0), "var in x,y,z,axangle"
    sens_noise_cov_mat = np.cov(pose_marker.T)
    ax.scatter(pose_marker[:,0],pose_marker[:,1],pose_marker[:,2])
np.savetxt("/home/biorobotics/Desktop/tejas/cpp_test/workingCodes/noise_filter/sens_noise_cov_mat.txt",sens_noise_cov_mat)

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

