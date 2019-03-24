import numpy as np 
import cv2

blank = np.zeros((600,600),dtype = np.uint8)
while (True):
	cv2.imshow("blank",blank)

	if cv2.waitKey(0) == ord('r'):
		print "recording"
	 

	if cv2.waitKey(0) & 0xFF == ord('q'):
		print "stopped"
		break


