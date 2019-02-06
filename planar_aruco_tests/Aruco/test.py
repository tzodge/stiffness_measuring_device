import cv2
from cv2 import aruco
#import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 

import random

import time
length = 15

dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
cap = cv2.VideoCapture(0)


with np.load('B.npz') as X:  # B.npz stores the calibration data
 	mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

R = np.zeros((3,3))
A = [-37.44,-115.26,-22.26]
CameraFrame = np.zeros((3,1))
TipPoseInMarkerFrame = np.transpose(A)



# plt.ion()
fig = plt.figure()
while True:
	ret, frame = cap.read()
	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = frame
	res = aruco.detectMarkers(gray, dictionary) 
#res = contains array of corners of detected markers, ID of markers, rejected points

	i = 0
	if len(res[0]) > 0:
		rvec, tvec, _ = aruco.estimatePoseSingleMarkers(res[0], length, mtx, dist)
		# print tvec.shape	
		# print "aa gaya"

		
		aruco.drawDetectedMarkers(gray, res[0],res[1])
		x = 0
		for x in xrange(0,len(rvec)):
			# print ("x = " ,x)
			a = res[1]
			print a
			# print type(rvec)
			print rvec.shape
			if rvec.shape == (10,1,3):
				# print random.random()
				
				print rvec.shape
				if a.item(x) == 9: #detects marker number 7
					aruco.drawAxis(gray,mtx,dist,rvec[x],tvec[x],length)
					# cv2.Rodrigues(rvec, R)
					print "aa gaya"
					# print np.transpose(rvec)
					# print rvec.shape	
					# poseInCameraFrame = np.dot(R,TipPoseInMarkerFrame) #+  np.transpose(tvec)
					# poseInCameraFrame = poseInCameraFrame.reshape((3,1)) + np.transpose(tvec[0])
					# print (poseInCameraFrame)
					# # print "aa gaya  "
					# print (np.transpose(tvec[0]).shape)
					# fig = plt.figure()		
					# ax = fig.add_subplot(111, projection ='3d')
					# ax.scatter3D(poseInCameraFrame[0],poseInCameraFrame[1],poseInCameraFrame[2] )
					# print "aa gaya"	
					# plt.pause(0.00005)
					# time.sleep(0.005)
					# plt.show()
					
					

		cv2.imshow('frame',gray)

   	 	if cv2.waitKey(1) & 0xFF == ord('q'):
			break

cap.release()
cv2.destroyAllWindows()