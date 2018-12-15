#Used this code to confirm that the tvec and rvec given by the 
 # estimatePoseSingleMarkers is of the marker frame wrt the camera frame


import numpy as np
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
from mpl_toolkits.mplot3d import Axes3D
import time

with np.load('B4Aruco.npz') as X:
    # mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
    mtx, dist = [X[i] for i in ('mtx','dist')] 


cap = cv2.VideoCapture(0)

iterations_for_while = 200
t = np.arange(iterations_for_while)
# aruco.CORNER_REFINE_SUBPIX

xs,ys,zs = [0]*iterations_for_while,[0]*iterations_for_while,[0]*iterations_for_while
j = 0
# plt.figure()
t0 = time.time() 
time_vect = [0]*iterations_for_while
# time_vect.append(0)
while(j<iterations_for_while):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #print(frame.shape) #480x640
    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # aruco.CORNER_REFINE_SUBPIX
    # print aruco.doCornerRefinemofdsjogidsjent
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    parameters =  aruco.DetectorParameters_create()
    parameters.cornerRefinementMethod = 1


    # print parameters.cornerRefinementMethod, "......................"
    # print getParameters()
 
    #print(parameters)
 
    '''    detectMarkers(...)
        detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
        mgPoints]]]]) -> corners, ids, rejectedImgPoints
        '''
        #lists of ids and the corners beloning to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    # print corners
    emptyList = list()
    emptyList.append(np.zeros((4,4)))

    if ids is not  None :
        for i in range(0,len(ids)):
            if ids[i] ==7:  
                emptyList[0] = corners[i]
                frame = aruco.drawDetectedMarkers(frame, emptyList)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers( emptyList, 19.16, mtx,dist)

    # print tvecs[0,0,0]
    # print 
    time_vect[j] = j 
    xs[j] = tvecs[0,0,0]
    ys[j] = tvecs[0,0,1]
    zs[j] = tvecs[0,0,2]
 
    j = j+1
    print j
    # plt.scatter (time_vect,xs,color= 'red')
    # plt.scatter (time_vect,ys,color= 'green')
    # plt.scatter (time_vect,zs,color= 'blue')
    # # plt.show()
    # plt.grid(linewidth=1)
    #                 # ax.set_yticks(np.arange(-120,41,10))
    # plt.pause(0.05)
    # cv2.imshow('frame',frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    
# When everything done, release the capture
print np.std(xs)
print np.std(ys)
print np.std(zs)
np.savetxt("x_coordinate_along_straight_line",xs,delimiter=',')
np.savetxt("y_coordinate_along_straight_line",ys,delimiter=',')
np.savetxt("z_coordinate_along_straight_line",zs,delimiter=',')
np.savetxt("all_coordinate_along_straight_line",np.array([xs,ys,zs]).T,delimiter=',')

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.scatter(xs,ys,zs,c=t)
ax.xlim = (-35 ,-45)
ax.ylim = (-30 ,20)
ax.zlim = (300 ,400)
# plt.axis('equal')

plt.show()



cap.release()
cv2.destroyAllWindows()