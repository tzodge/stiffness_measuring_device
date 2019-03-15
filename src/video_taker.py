#Used this code to confirm that the tvec and rvec given by the 
 # estimatePoseSingleMarkers is of the marker frame wrt the camera frame


# from __future__ import division
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

 
j = 0  # iteration counter


rospy.init_node('RosCam', anonymous=True)
ic = RosCam("/camera/image_color")

t_prev = time.time()   	
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 30.0, (1280,1024))
while(True):
	t_new = time.time()   
	frame = ic.cv_image
 
	if frame is None:
		time.sleep(0.1)
		print("No image")
		continue
	else:
		frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)		
		out.write(frame) 
		print("frame number ", j)
		cv2.imshow('frame_color',frame)
		print("current frabme rate",1./(t_new- t_prev))
		j+=1
		if cv2.waitKey(0) & 0xFF == ord('q') :
			print "stopped"
			break

	t_prev = t_new
	 

out.release()
cv2.destroyAllWindows()



