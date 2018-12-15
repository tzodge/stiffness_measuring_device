import numpy as np 
import cv2 as cv
from cv2 import aruco
import matplotlib.pyplot as plt

with np.load('B.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

print mtx    
cap = cv.VideoCapture(0)
cv.namedWindow("vid")
b = np.empty((1,))
c = np.empty((1,))
d = np.empty((1,))
while (True):
	ret, frame = cap.read()
	

	
	aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
	parameters = aruco.DetectorParameters_create()
	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters = parameters)
	# print corners
	gray = aruco.drawDetectedMarkers(gray,corners)
	i = 0
	if type(ids) == type(np.empty((2,3))):

		for x in range (0,len(ids)):
			if ids[x] == 8:
				rvec, tvec,_ = aruco.estimatePoseSingleMarkers(corners[x], 18.96, mtx, dist)
				
				b= np.append (b,tvec[0][0,0])
				print tvec[0][0,0]
				print "..."
				 
				
				c= np.append (c,tvec[0][0,1])
				# print c
				print tvec[0][0,1]
				print "..."
				
				d= np.append (d,tvec[0][0,2])
				# print c
				print tvec[0][0,2]
				print "..."
				
	

				# d= np.append (d,tvec[0][0,2])
				
				# print b.shape
				# print c.shape
				



	cv.imshow("vid",gray)
	k = cv.waitKey(1)
	if k%256 == 27:
		print("Escape hit, closing...")
		break


cap.release()
cv.destroyAllWindows()
b = np.array(b[1:])

c = np.array(c[1:])
d = np.array(d[1:])

print b
print ".."
print c
print "..."
print d
print "..."

plt.plot(b,c,'ro')

plt.show()


print np.std(b)
print ""
print np.std(c)
print ""
print np.std(d)
print ""