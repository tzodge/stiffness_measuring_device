import numpy as np
import cv2

img  = np.zeros([300,300,3])
# img = cv2.CreateImage((300,300),8,3)

meshgrid,_ = np.mgrid[0:300,0:300]
img[:,:,0] = meshgrid
img[:,:,1] = meshgrid
img[:,:,2] = meshgrid
img = img.astype(np.uint8)
frame = cv2.imread("sample_image_aruco.jpg")
frame_2 = np.copy(frame)
print type(img),"type(img)"
print type(frame),"type(frame)"

print img.shape,"img.shape"
print frame.shape,"frame.shape"

print frame.dtype, "frame.dtype"	
cv2.imwrite("img.jpg",img)
img_again = cv2.imread("img.jpg",)



cv2.imshow("frame_2",frame_2)
cv2.imshow("img_again",img_again)
cv2.imshow("img",img)
cv2.waitKey(0)