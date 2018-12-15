
#Used this code to confirm that the tvec and rvec given by the 
 # estimatePoseSingleMarkers is of the marker frame wrt the camera frame


import numpy as np
import cv2 as cv
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
with np.load('B4Aruco.npz') as X:
    # mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
    mtx, dist = [X[i] for i in ('mtx','dist')] 


def ID2toID1Estimate (transf1, transf2):
	transf = np.matmul(np.linalg.inv(transf1), transf2)	
	rotmtx = (transf[0:3,0:3])
	rvec,_ = cv.Rodrigues(rotmtx)
	tvec = np.reshape((transf[0:3,-1]),(3,1))
	X = np.vstack ((rvec,tvec))
	return X


def col(x):
	x = x.reshape(len(x[0]),1)
	return x

def RodriguesToTransf(x):
	x = np.array(x)
	# print x[0:3]
	
	rot,_ = cv.Rodrigues(x[0:3])
	
	
	trans =  np.reshape(x[3:6],(3,1))

	Trransf = np.concatenate((rot,trans),axis = 1)
	Trransf = np.concatenate((Trransf,np.array([[0,0,0,1]])),axis = 0)

	return Trransf

sub_pix_refinement_switch = 1
counter_lim = 100
counter = 0
ImageCount = np.array([[0]])
X_error = np.array([[0]])
Y_error = np.array([[0]])
Z_error = np.array([[0]])
X_measured = np.array([[0]])
cap = cv.VideoCapture(0)


 
while(counter < counter_lim):
    # Capture frame-by-frame
    ret, frame = cap.read()
   

    #print(frame.shape) #480x640
    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    parameters =  aruco.DetectorParameters_create()
    parameters.cornerRefinementMethod = sub_pix_refinement_switch

    # print parameters , "parameters"
    #print(parameters)
 
    '''    detectMarkers(...)
        detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
        mgPoints]]]]) -> corners, ids, rejectedframePoints
        '''
        #lists of ids and the corners beloning to each id
    corners, ids, rejectedframePoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    # print(corners)
    print "in while loop"
 
    emptyList = list()
    emptyList.append(np.zeros((4,4)))
    emptyList.append(np.zeros((4,4)))

    if ids is not  None :
        if len(corners) >= 2:
        	# print "is >2"

		for j in range(0, len(ids)):
			# print ids
							
			for k in range (0, len(corners)):
				if (ids[j][0] == 6 and ids[k][0] ==7):
					emptyList[0] = corners[j]
					emptyList[1] = corners[k]
					# print "batabta"	
					
					aruco.drawDetectedMarkers(frame,emptyList)
					rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers( corners, 21.5, mtx,dist)
					
					if ids[j] > ids[k]:
						a = "{}".format(ids[k][0]) + "R" +"{}".format(ids[j][0]) #KeyName 
						tr1 = np.reshape(np.append(rvecs[j],tvecs[j]),(6,1))
						tr2 = np.reshape(np.append(rvecs[k],tvecs[k]),(6,1))
						TR1 = RodriguesToTransf(tr1)
						TR2 = RodriguesToTransf(tr2)
						TR12 =  np.matmul(np.linalg.inv(TR2),TR1)
					
						# MeasuredRelativePoses.update( {a : TR12})
						# transformationKey.update({a: np.array(([ids[k], ids[j]]))})

					else :
						
						a = "{}".format(ids[j][0]) + "R" +"{}".format(ids[k][0]) #KeyName 
						tr1 = np.reshape((np.append(rvecs[k],tvecs[k])),(6,1))
						tr2 = np.reshape((np.append(rvecs[j],tvecs[j])),(6,1))
						
						TR1 = RodriguesToTransf(tr1)
						TR2 = RodriguesToTransf(tr2)
						TR12 =  np.matmul(np.linalg.inv(TR1),TR2)
						# MeasuredRelativePoses.update( {a : TR12})
						# transformationKey.update({a: np.array(([ids[j], ids[k]]))})
					# print "measured"
			
					
		
# checking the images and number of detected markers
	# print np.array([[TR12[0,3]]])
					# X_given = 51.8 
					# ImageCount = np.concatenate((ImageCount,np.array([[counter]])), axis = 1 )   				
					# X_error = np.concatenate((X_error,np.array([[TR12[0,3]]])), axis = 1 )
					# Y_error = np.concatenate((Y_error,np.array([[TR12[1,3]]])), axis = 1 )
					# Z_error = np.concatenate((Z_error,np.array([[TR12[2,3]]])), axis = 1 )
					# X_measured = np.concatenate((X_measured,np.array([[X_given]])), axis = 1 )  
					# print ImageCount
					# print (ImageCount[0])
					# print (X_error[0])
					# print counter
					
					# plt.plot (ImageCount[0],X_error[0],'r')
					# plt.plot (ImageCount[0],Y_error[0],'g')
					# plt.plot (ImageCount[0],Z_error[0],'b')
					# print TR12[0,3]
					print counter
					plt.scatter(counter,TR12[0,3], c = 'r')
					plt.scatter(counter,TR12[1,3], c = 'g')
					plt.scatter(counter,TR12[2,3], c = 'b')
					# plt.tight_layout()
					print 	TR12[1,3], "TR12[1,3]"
					print 	TR12[2,3], "TR12[2,3]"

					# plt.hold(True)					

					# plt.plot (ImageCount[0],X_measured[0],'black')
					
					plt.grid(linewidth=1) 
					plt.pause(0.05)
					# ax.set_yticks(np.arange(-120,41,10))
					# plt.ion()
					
					wait = 10
					cv.imshow('image',frame)
					k = cv.waitKey(wait)
					if k%256 == 32:
						break 
	counter = counter +1 
 
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()