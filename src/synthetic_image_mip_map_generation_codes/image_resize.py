import cv2
import os
import numpy as np

def draw_masks (frame):


# def marker_edges (marker_id, dwn_smpl):
	
	# marker_id_str = "aruco_images" + "/{}.jpg".format(marker_id)  
	# gray = cv2.imread(marker_id_str)
	print frame.shape,"frame.shape"

	frame_gray_tmp = np.copy(frame)
	cv2.imshow("frame",frame)
	cv2.waitKey(0)
	# print frame.shape,"frame.shape"
	frame_gray = cv2.cvtColor(frame_gray_tmp,cv2.COLOR_BGR2GRAY)
	# a = cv2.Canny(frame_gray,100,200,apertureSize = 3)

	# a[0,:] = 255
	# a[-1,:] = 255
	# a[:,0] = 255
	# a[:,-1] = 255
	cv2.imshow("image",frame_gray)
# 	a = np.fliplr(a.T) #do not change this!!! #            
# 	scale = a.shape[0]/marker_size_in_mm
# 	a_shift = a.shape[1]/(2*scale)
# 	b = np.asarray(np.nonzero(a))/scale - a_shift
# 	z = np.zeros((1,b.shape[1]))
# 	tr_dash = np.ones((1,b.shape[1]))
# 	b = np.vstack ((b,z,tr_dash))       
# 	interval =  int(b.shape[1]/dwn_smpl)
# #    print (interval,"interval " )
# 	b = b[:,0::interval]
# 	return b



path = "aruco_images/"
path_save = "aruco_images_mip_maps/"
for img in os.listdir(path):
	print img

	frame = cv2.imread(path+img)
	frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	frame_size = frame_gray.shape[0]
	frame_size_with_border = int(1.2*frame_size)
	start_ind = int(frame_size*0.1)
	end_ind = int(frame_size*1.1)

	frame_with_border = np.ones((frame_size_with_border,frame_size_with_border))
	frame_with_border = frame_with_border.astype(np.uint8)
		
	# frame_with_border = np.ones((frame_size_with_border,frame_size_with_border,3))
	frame_with_border[:,:] = 255
	# frame_with_border[:,:,1] = 255
	# frame_with_border[:,:,2] = 255

	# print frame_with_border[start_ind:end_ind,start_ind:end_ind,:].shape	
	# print frame.shape
	frame_with_border[start_ind:end_ind,start_ind:end_ind] = frame_gray
	print start_ind,"start_ind"
	print end_ind,"end_ind"
	# draw_masks(frame_with_border)

	res_300 = cv2.pyrDown(frame_with_border)
	res_150 = cv2.pyrDown(res_300)
	res_75 = cv2.pyrDown(res_150)
	res_38 = cv2.pyrDown(res_75)

	res_300 = cv2.pyrUp(res_300)
	res_150 = cv2.pyrUp(cv2.pyrUp(res_150))
	res_75 = cv2.pyrUp(cv2.pyrUp(cv2.pyrUp(res_75)))
	res_38 = cv2.pyrUp(cv2.pyrUp(cv2.pyrUp(cv2.pyrUp(res_38))))


	cv2.imshow("res_300",res_300)
	cv2.imshow("res_150",res_150)
	cv2.imshow("res_75",res_75)
	cv2.imshow("res_38",res_38)

	# cv2.imshow("frame_with_border",frame_with_border)
	# cv2.imshow("frame_gray",frame_gray)

	cv2.imwrite(path_save + "res_38_" + img, res_38)
	cv2.imwrite(path_save + "res_75_" + img, res_75)
	cv2.imwrite(path_save + "res_150_" + img, res_150)
	cv2.imwrite(path_save + "res_300_" + img, res_300)
	cv2.imwrite(path_save + "res_600_" + img, frame_with_border)
	cv2.waitKey(0)


