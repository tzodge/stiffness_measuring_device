#Used this code to confirm that the tvec and rvec given by the 
 # estimatePoseSingleMarkers is of the marker frame wrt the camera frame


import numpy as np
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
from mpl_toolkits.mplot3d import Axes3D
import transforms3d as tf3d
import time
import glob		
from scipy.optimize import minimize
import matplotlib.mlab as mlab

def RodriguesToTransf(x):
    #input (6,)
    x = np.array(x)
    rot,_ = cv2.Rodrigues(x[0:3])
    trans =  np.reshape(x[3:6],(3,1))
    Trransf = np.concatenate((rot,trans),axis = 1)
    Trransf = np.concatenate((Trransf,np.array([[0,0,0,1]])),axis = 0)
    return Trransf


def obj_func(xyz_gs):

	### x_gs,y_gs,z_gs are initial guesses
	predicted_loc = [0]*valid_img_num

	gs_vect = np.array([xyz_gs[0],xyz_gs[1],xyz_gs[2],1])
	gs_vect = gs_vect.reshape(4,1)

	for i in range(valid_img_num):
		predicted_loc[i] = transf_mat_for_img[i].dot(gs_vect)
	b = np.std(predicted_loc,axis=0)	

	# for i in range(valid_img_num):
			

	return b.sum()

print ""
### Switches: 
sub_pix_refinement_switch = 1
num_img_considered = 1000
marker_size_in_mm = 19.16
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters =  aruco.DetectorParameters_create()
parameters.cornerRefinementMethod = sub_pix_refinement_switch
total_num_of_img = 1000
transf_mat_for_img = [0]*total_num_of_img 
valid_img_num = 0 
initial_guess = np.array([3.0 ,-114.0,   -6.0])
# mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
with np.load('B4Aruco.npz') as X:
	# print "fasdfd"
	mtx, dist = [X[i] for i in ('mtx','dist')]





for img in glob.glob('data_images/*.jpg'):
	img = cv2.imread(img)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=parameters)

	if ids is not None:

		aruco.drawDetectedMarkers(img, corners)	
		print valid_img_num
		rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size_in_mm, mtx,dist)
		transf_mat_for_img[valid_img_num] = RodriguesToTransf(np.append(rvecs,tvecs))
		valid_img_num += 1 
		# cv2.imshow('image',img)
		# cv2.waitKey()
	if valid_img_num >= num_img_considered:
		break
print "valid_img_num",valid_img_num


del(transf_mat_for_img[valid_img_num:])


# print obj_func(np.array([10,-115,-10]))

res = minimize(obj_func, np.array(initial_guess), method='nelder-mead',options={'xtol': 1e-8, 'disp': True})

print res.x,"result "
print ""
a = res.x

b= np.array([a[0],a[1],a[2],1])
c = b.reshape(4,1)

predicted_loc = [0]*valid_img_num
predicted_loc_x = [0]*valid_img_num
predicted_loc_y = [0]*valid_img_num
predicted_loc_z = [0]*valid_img_num

for i in range(valid_img_num):
		predicted_loc[i] = transf_mat_for_img[i].dot(c)
		# print predicted_loc[i][0,0]
		# print predicted_loc[i][1,0]
		predicted_loc_x[i] = predicted_loc[i][0,0] 
		predicted_loc_y[i] = predicted_loc[i][1,0] 
		predicted_loc_z[i] = predicted_loc[i][2,0] 
		# print "........"


plt.hist(predicted_loc_z,num_img_considered/10,normed = 1,facecolor='blue')
plt.hist(predicted_loc_x,num_img_considered/10,normed = 1,facecolor='red')
plt.hist(predicted_loc_y,num_img_considered/10,normed = 1,facecolor='green')

plt.show()

mean_pred_loc = np.mean(predicted_loc, axis=0)
std_dev_pred_loc =np.std(predicted_loc, axis=0)

noisy_measurement_list = []
for i in range (valid_img_num):
	if np.linalg.norm(predicted_loc[i][0:3,0]- mean_pred_loc[0:3,0]) >= np.linalg.norm(std_dev_pred_loc[0:3,0]): 
		noisy_measurement_list.append(i)


for i in range(valid_img_num):	
	print predicted_loc[i], i
	print ""

for i in sorted (noisy_measurement_list , reverse =True):
	del predicted_loc[i]
	del transf_mat_for_img[i]


valid_img_num = len(transf_mat_for_img)
print mean_pred_loc,"mean_pred_loc"
print std_dev_pred_loc,"std_dev_pred_loc"

# print len(predicted_loc)


print noisy_measurement_list
# print len(transf_mat_for_img)





res = minimize(obj_func, np.array(res.x), method='nelder-mead',options={'xtol': 1e-8, 'disp': True})

print res.x,"result "
print ""
a = res.x





print valid_img_num







# print predicted_loc

# print predicted_loc
# b = np.std(predicted_loc,axis=1)
# print obj_func(res.x)

# num = 100
# x_plot = np.linspace(-50,50,num)
# y_plot = np.linspace(0,15,num)
# z_plot = np.linspace(200,300,num)
# color = [0]*num

# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')


# for i in range (num):
# 	color[i] = obj_func([x_plot[i],y_plot[i],z_plot[i]])	
# 	# color = ax.scatter(x_plot[i],y_plot[i],z_plot[i])
# 			# print i,"",j,"",k

# # print predicted_loc

# ax.scatter(x_plot,y_plot,z_plot,c=color)

# plt.show()

# xs,ys,zs,color = [0]*iterations_for_while,[0]*iterations_for_while,[0]*iterations_for_while,[0]*iterations_for_while


# j = 0

# t0 = time.time() 
# time_vect = [0]*iterations_for_while

# while(j<iterations_for_while):
#     # Capture frame-by-frame
#     ret, frame = cap.read()


 
#     #print(parameters)
 
#     '''    detectMarkers(...)
#         detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
#         mgPoints]]]]) -> corners, ids, rejectedImgPoints
#         '''
#         #lists of ids and the corners beloning to each id
#     corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
#     # print corners
#     emptyList = list()
#     emptyList.append(np.zeros((4,4)))

#     if ids is not  None :
#         for i in range(0,len(ids)):
#             if ids[i] ==7:  
#                 emptyList[0] = corners[i]
#                 frame = aruco.drawDetectedMarkers(frame, emptyList)
#                 rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers( emptyList, marker_size_in_mm, mtx,dist)

#     # print tvecs[0,0,0]
#     # print 
#     xs[j] = tvecs[0,0,0]
#     ys[j] = tvecs[0,0,1]
#     zs[j] = tvecs[0,0,2]
#     color[j] = j 
#     time_vect[j] = time.time() - t0
#     rot,_ = cv2.Rodrigues(rvecs)


    


#     j = j+1
#     print j
#     # plt.scatter (time_vect,xs,color= 'red')
#     # plt.scatter (time_vect,ys,color= 'green')
#     # plt.scatter (time_vect,zs,color= 'blue')
#     # # plt.show()
#     # plt.grid(linewidth=1)
#     #                 # ax.set_yticks(np.arange(-120,41,10))
#     # plt.pause(0.05)
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q') or j >= iterations_for_while:
#         break

# cap.release()
# cv2.destroyAllWindows()

# ## removing list elements whicj didn't update
# del(xs[j:])  
# del(ys[j:])  
# del(zs[j:])  
# del(color[j:])
# del(time_vect[j:])

# # When everything done, release the capture
# print np.std(xs)
# print np.std(ys)
# print np.std(zs)

# ###Saving the data
# # np.savetxt("x_coordinate_along_straight_line",xs,delimiter=',')
# # np.savetxt("y_coordinate_along_straight_line",ys,delimiter=',')
# # np.savetxt("z_coordinate_along_straight_line",zs,delimiter=',')
# # np.savetxt("time_vect",time_vect,delimiter=',')
# # np.savetxt("all_coordinate_along_straight_line",np.array([xs,ys,zs]).T,delimiter=',')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.scatter(xs,ys,zs,c=color)
# ax.xlim = (-35 ,-45)
# ax.ylim = (-30 ,20)
# ax.zlim = (300 ,400)
# # plt.axis('equal')

# plt.show()

# # 