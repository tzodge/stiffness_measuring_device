
## est_pos_marker = sensor data
## pred_pos_marker = prediction using velocity model

#### won't work because the velocities you are finding to predict the positions is also errorenous
##### this results in accumulating the error and ends up in giving worse results



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


with np.load('B4Aruco.npz') as X:
	mtx,dist = [X[i] for i in ('mtx','dist')]


### constants
iterations_for_while = 200
marker_size_in_mm = 19.16
tip_coord  = np.array([  3.13173022, -112.71652409 ,  -7.28093744,1]) 
std_error = 0.5

### Switches: 
sub_pix_refinement_switch = 1
detect_tip_switch = 0
filter_switch = 0




xs,ys,zs,color = [0]*iterations_for_while,[0]*iterations_for_while,[0]*iterations_for_while,[0]*iterations_for_while


cap = cv2.VideoCapture(0)
est_pos_marker = np.zeros((iterations_for_while,3))
vel_marker = np.zeros((iterations_for_while,3))
pred_pos_marker = np.zeros((iterations_for_while,3))
norm_diff_est_pred = np.zeros((iterations_for_while,1))
updated_pos_marker = np.zeros((iterations_for_while,3))



aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters =  aruco.DetectorParameters_create()
parameters.cornerRefinementMethod = sub_pix_refinement_switch
parameters.cornerRefinementMinAccuracy = 0.05

j = 0

t0 = time.time() 
time_vect = [0]*iterations_for_while

while(j<iterations_for_while):
    # Capture frame-by-frame
    ret, frame = cap.read()
    time_vect[j] = time.time()-t0




 
    #print(parameters)
 
    '''    detectMarkers(...)
        detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
        mgPoints]]]]) -> corners, ids, rejectedImgPoints
        '''
        #lists of ids and the corners beloning to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    emptyList = list()
    emptyList.append(np.zeros((4,4)))

    if ids is not  None :
        for i in range(0,len(ids)):
            if ids[i] ==7:  
                emptyList[0] = corners[i]
                # print corners[i]
                frame = aruco.drawDetectedMarkers(frame, emptyList)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers( emptyList, marker_size_in_mm, mtx,dist)
                transf_mat_for_frame = RodriguesToTransf(np.append(rvecs,tvecs))
    # print tvecs[0,0,0]
    # print 
        # print tip_loc_cam_frame


    if detect_tip_switch ==1:
        tip_loc_cam_frame = transf_mat_for_frame.dot(tip_coord.T)
        xs[j] = tip_loc_cam_frame[0]
        ys[j] = tip_loc_cam_frame[1]
        zs[j] = tip_loc_cam_frame[2]
        
    else:        
        est_pos_marker[j,:] = tvecs[0,0,:]

	updated_pos_marker[0,:] = est_pos_marker[0,:]	
	updated_pos_marker[1,:] = est_pos_marker[1,:]	
	
	if filter_switch == 1:
 		if j > 1:
 			vel_marker[j-1,:] = (est_pos_marker[j-1,:]-est_pos_marker[j-2,:])/(time_vect[j-1]-time_vect[j-2])
 			pred_pos_marker[j,:] = vel_marker[j-1,:]*(time_vect[j]-time_vect[j-1]) + est_pos_marker[j-1,:]
 			norm_diff_est_pred[j,0] = np.linalg.norm(pred_pos_marker[j,:] - est_pos_marker[j,:])
 			if  norm_diff_est_pred[j,0]> std_error:
 				updated_pos_marker[j,:] = pred_pos_marker[j,:] 
 				print "."
 			else:
 				updated_pos_marker[j,:] = est_pos_marker[j,:] 

 			  



    color[j] = j 
    time_vect[j] = time.time() - t0
    rot,_ = cv2.Rodrigues(rvecs)

    j = j+1
    print j

    plt.pause(0.05)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or j >= iterations_for_while:
        break

cap.release()
cv2.destroyAllWindows()

if detect_tip_switch == 1:
	print np.std(updated_pos_marker[:,0]), "std_x"
	print np.std(updated_pos_marker[:,1]), "std_y"
	print np.std(updated_pos_marker[:,2]), "std_z"
else: 
	print np.std(est_pos_marker[:,0]), "std_x"
	print np.std(est_pos_marker[:,1]), "std_y"
	print np.std(est_pos_marker[:,2]), "std_z"



fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
# ax.scatter(updated_pos_marker[:,0],updated_pos_marker[:,1],updated_pos_marker[:,2],c=color  )
ax.scatter(est_pos_marker[:,0],est_pos_marker[:,1],est_pos_marker[:,2],c=color  )


# fig = plt.figure()
# plt.plot(time_vect,vel_marker[:,0], color='red')
# plt.plot(time_vect,vel_marker[:,1], color='green')
# plt.plot(time_vect,vel_marker[:,2], color='blue')

# fig = plt.figure()
# plt.plot(time_vect,est_pos_marker[:,0],'o' ,color='red')
# plt.plot(time_vect,est_pos_marker[:,1],'o' ,color='green')
# plt.plot(time_vect,est_pos_marker[:,2],'o' ,color='blue')

# print vel_marker, "vel_marker"
# print "..."
# print est_pos_marker, "est_pos_marker"
# print norm_diff_est_pred 
plt.show()