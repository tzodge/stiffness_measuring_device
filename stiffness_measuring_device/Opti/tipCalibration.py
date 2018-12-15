import numpy as np
import cv2 as cv
from cv2 import aruco



X = np.load("PoseWrt1.npy")


PoseWrt1 = X.item() 
print PoseWrt1["1R8"]
print "................"

def RodriguesToTransf(x):
	x = np.array(x)
	
	
	rot,_ = cv.Rodrigues(x[0:3])
	
	
	trans =  np.reshape(x[3:6],(3,1))

	Trransf = np.concatenate((rot,trans),axis = 1)
	Trransf = np.concatenate((Trransf,np.array([[0,0,0,1]])),axis = 0)
	# print Trransf
	return Trransf

def TransfToRodrigues(x):
	RotMat = x[0:3,0:3]
	RotMat,_ = cv.Rodrigues(RotMat)
	t = np.reshape(x[0:3,3],(3,1))
	
	Y = np.vstack((RotMat,t))
	print Y
	return Y



#estimated tip pose wrt ID1 = (-37.42, -115.257, -22.258)

NoOfImages = 41
with np.load('B3Aruco.npz') as X:
	# mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
	mtx, dist = [X[i] for i in ('mtx','dist')]
# Arrays to store object points and image points from all the images.


print mtx
print ""

path = "/home/biorobotics/cpp_test/Opti/penTipCalibData"
# MeasuredRelativePosesM = {} #relative measured poses willbe stored here
# MeasuredRelativePosesV = {}  
# transformationKey = {}
# CorrectedPose = {}
# PoseCount  = {}
positionOfTipEstimated = np.reshape(np.array([-37.42, -115.257, -22.258, 1]) , (4,1)) 
for i in range (0,NoOfImages):
	fname = "penTipCalibData" + "/opencv_frame_{}".format(i) + ".jpg"
	# print fname
	img = cv.imread(fname)
	aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
	parameters = aruco.DetectorParameters_create()
	
	corners, ids, _ = aruco.detectMarkers(img, aruco_dict, parameters = parameters)
	aruco.drawDetectedMarkers(img,corners,ids)
	rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers( corners, 19.16, mtx,dist)
	j = 0
	key = "1R{}".format(ids[j][0])
	print key
	# print PoseWrt1[key]
	CamTRID = RodriguesToTransf(np.reshape((np.append(rvecs[j],tvecs[j])),(6,1)))
	TipInCameraFrame = np.linalg.multi_dot((CamTRID,PoseWrt1[key],positionOfTipEstimated))
	print TipInCameraFrame



	
	cv.imshow("image", img)
	k = cv.waitKey(0)
	if k%256 == 32:
		break 

 