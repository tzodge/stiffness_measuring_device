import cv2
import numpy as np
import time
from matplotlib.path import Path
import matplotlib.pyplot as plt
 



## reference https://stackoverflow.com/questions/21339448/how-to-get-list-of-points-inside-a-polygon-in-python

def draw_patch(frame,corners_pix,bounding_box):
	t0 = time.time() 
	a = bounding_box[0] 
	b = bounding_box[1] 
	start_pnt = [b[0],a[0]]
	 

	x, y = np.meshgrid(a, b) # make a canvas with coordinates
	x, y = x.flatten(), y.flatten()
	points = np.vstack((x,y)).T 

	p = Path(corners_pix) # make a polygon in pixel space
	grid = p.contains_points(points)  # make grid
	mask = grid.reshape(len(a),len(b)) 
	local_frame = frame[start_pnt[0]:start_pnt[0]+len(a), start_pnt[1]:start_pnt[1]+len(b)]
	local_frame_grad,_ = np.gradient(local_frame)
	local_frame_grad_int8 = np.asarray(local_frame_grad,dtype=np.uint8)
	np.copyto(frame[start_pnt[0]:start_pnt[0]+len(a), start_pnt[1]:start_pnt[1]+len(b)],
		local_frame_grad_int8,where=mask)

	t1 = time.time() 
	print t1-t0,"for func"
	 
	# cv2.imshow("frame",frame)
	# cv2.imshow("mask",np.array(mask*1*255,dtype=np.uint8))
	return frame

# frame = np.random.randint(255,size=[480,640],dtype = np.uint8)
frame = cv2.imread("send_to_howard.png",0)
corners_pix = np.array([[200,300],[300,400], [500,300],[400,200]  ])  
a = np.arange(0,300)+0 
b = np.arange(100,400)+50 
print a[0].shape
bounding_box = [a,b]  
frame_new  =  draw_patch(frame,corners_pix,bounding_box)

for i in range(len(corners_pix)	):
	cv2.circle(frame_new,tuple(corners_pix[i,:]),20,5,2)

cv2.circle(frame_new,tuple([a[0],b[0]]),10,127,2)
cv2.circle(frame_new,tuple([a[-1],b[0]]),10,127,2)
cv2.circle(frame_new,tuple([a[0],b[-1]]),10,127,2)
cv2.circle(frame_new,tuple([a[-1],b[-1]]),10,127,2)
 

cv2.imshow("frame_new",frame_new)
cv2.waitKey(0)	
