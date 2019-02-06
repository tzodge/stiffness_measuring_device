import cv2 as cv
import numpy as np 
from cv2 import aruco
import glob
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




TransfMatIDtoCent = np.load("TransfMatIDtoCent.npy") 


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




NoOfImages = 12
with np.load('B5Aruco.npz') as X:
	# mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
	mtx, dist = [X[i] for i in ('mtx','dist')]

# Arrays to store object points and image points from all the images.


print mtx

print ""
path = "/home/biorobotics/cpp_test/Opti/deleteafterwards/MarkerInsidePentagonSet_1_Glossypaper"
MeasuredRelativePoses = {}  #relative measured poses willbe stored here
transformationKey = {}
CorrectedPose = {}
PoseCount  = {}

ImageCount = np.array([[0]])
X_error = np.array([[0]])
Y_error = np.array([[0]])
Z_error = np.array([[0]])
X_measured = np.array([[0]])

fig = plt.figure()
for i in range (0,NoOfImages):
	fname = path + "/opencv_frame_{}".format(i) + ".jpg"
	
	img = cv.imread(fname)
	# cv.imshow("image",img)
	aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
	parameters = aruco.DetectorParameters_create()
	# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	corners, ids, _ = aruco.detectMarkers(img, aruco_dict, parameters = parameters)
	rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers( corners, 21.5, mtx,dist)
	


####plotting the location of the markers wrt camera
	X = np.zeros((1,21))
	Y = np.zeros((1,21))
	Z = np.zeros((1,21))
	for j in range (0,len(corners)-1):
		X[0,j] = tvecs[j,0,0]
		Y[0,j] = tvecs[j,0,1]
		Z[0,j] = tvecs[j,0,2]
		# print "translation from {} to {} = {}".format(ids[j],ids[j+1], tvecs[j,0]-tvecs[j+1,0])
		# print "Error in position = {}".format(np.linalg.norm(tvecs[j,0]-tvecs[j+1,0]))
		# print""
	# print X

	
	
	# ax = fig.add_subplot(111, projection='2d')	
	# ax.scatter(X,Y,Z) 
	# plt.show()

	
	if len(corners) >= 2:
		aruco.drawDetectedMarkers(img,corners,ids)
		print len(ids)	

		for j in range(0, len(ids)):
			# print "........................"
							
			for k in range (0, len(corners)):
				


				if (ids[j][0] == 11 and ids[k][0] ==10):
					
					if ids[j] > ids[k]:
						# R1,_ = cv.Rodrigues(rvecs[i]) 
						# R2,_ = cv.Rodrigues(rvecs[j])
						a = "{}".format(ids[k][0]) + "R" +"{}".format(ids[j][0]) #KeyName 
						tr1 = np.reshape(np.append(rvecs[j],tvecs[j]),(6,1))
						tr2 = np.reshape(np.append(rvecs[k],tvecs[k]),(6,1))
						TR1 = RodriguesToTransf(tr1)
						# print TR1
						
						TR2 = RodriguesToTransf(tr2)
						# print TR2
						
						TR12 =  np.matmul(np.linalg.inv(TR2),TR1)
						
						print a						
						if a in PoseCount:
							# print "first"
							# TR12 =  (TR12 + PoseCount[a]*MeasuredRelativePoses[a])/(PoseCount[a] + 1) #taking average of poses between same IDS in all frames 
							PoseCount[a] = PoseCount[a] + 1 

						else:
							# print "second"
							# R12 =  np.matmul(np.linalg.inv(R2),R1)
							PoseCount[a] = 1

						print a	
						# print TR1
						# print TR2 
						# print "."
						print "first"
						print TR12
			



						MeasuredRelativePoses.update( {a : TR12})
						transformationKey.update({a: np.array(([ids[k], ids[j]]))})

					else :
						# R1,_ = cv.Rodrigues(rvecs[j])
						# R2,_ = cv.Rodrigues(rvecs[i])
						a = "{}".format(ids[j][0]) + "R" +"{}".format(ids[k][0]) #KeyName 
						tr1 = np.reshape((np.append(rvecs[k],tvecs[k])),(6,1))
						tr2 = np.reshape((np.append(rvecs[j],tvecs[j])),(6,1))
						
						TR1 = RodriguesToTransf(tr1)
						TR2 = RodriguesToTransf(tr2)
						TR12 =  np.matmul(np.linalg.inv(TR1),TR2)
						
						


						if a in PoseCount:
							# print "third"
							# TR12 =  (TR12 + PoseCount[a]*MeasuredRelativePoses[a])/(PoseCount[a] + 1)
							PoseCount[a] = PoseCount[a] + 1 
						else:
							# print "fourth"
							# R12 =  np.matmul(np.linalg.inv(R2),R1)
							PoseCount[a] = 1

						MeasuredRelativePoses.update( {a : TR12})
						transformationKey.update({a: np.array(([ids[j], ids[k]]))})
						print a	
						# print TR1
						# print TR2 
						# print "."
						print "second"
						
						print ""
						print ""
					# print "measured"
			
					
		
# checking the images and number of detected markers
	# print np.array([[TR12[0,3]]])
	X_given = 51.8 
	ImageCount = np.concatenate((ImageCount,np.array([[i]])), axis = 1 )   				
	X_error = np.concatenate((X_error,np.array([[TR12[0,3]]])), axis = 1 )
	Y_error = np.concatenate((Y_error,np.array([[TR12[1,3]]])), axis = 1 )
	Z_error = np.concatenate((Z_error,np.array([[TR12[2,3]]])), axis = 1 )
	X_measured = np.concatenate((X_measured,np.array([[X_given]])), axis = 1 )  
	# print ImageCount
	plt.plot (ImageCount[0],X_error[0],'r')
	plt.plot (ImageCount[0],Y_error[0],'g')
	plt.plot (ImageCount[0],Z_error[0],'b')
	print X_measured
	print X_error.shape
	print X_measured.shape

	plt.plot (ImageCount[0],X_measured[0],'black')
	
	plt.grid(linewidth=1)
	# ax.set_yticks(np.arange(-120,41,10))
	plt.pause(0.05)
	plt.ion()
	print fname
	wait = 100
	if i == NoOfImages-1:
		wait = 0
	cv.imshow('image',img)
	k = cv.waitKey(wait)
	if k%256 == 32:
		break 
			




#optimiser
'''
EstimatedPose = np.empty((1,1), float)
ObservedPose = np.empty((1,1), float)
# print type(EstimatedPose)
for key in MeasuredRelativePoses.iterkeys():
	    X = ID2toID1Estimate(TransfMatIDtoCent[transformationKey[key][0,0]], TransfMatIDtoCent[transformationKey[key][1,0]])  
	    print key

	    EstimatedPose = np.vstack((EstimatedPose,X))
	    ObservedPose = np.vstack((ObservedPose,MeasuredRelativePoses[key]))


def ErrorFunc (RelPose):
	sum = 0
	for i in range (0,len(RelPose)):
		sum = sum + (RelPose[i] - ObservedPose[i])**2 

	return sum   





res = minimize(ErrorFunc, EstimatedPose,method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
X = res.x
np.save("CorrectedPosesArray", X)

'''









X = np.load("CorrectedPosesArray.npy")

i= 0
for key in MeasuredRelativePoses.iterkeys():

	CorrectedPose[key] = X[6*i +1 : 6*i +7]
	# print key
	# print CorrectedPose[key]

	# print "................."
	i = i+1
	# print ""

	    
'''we have to arrage the frames in descending order in a tree
so we will first rearrange the keys in the correctedPose
Meas
'''

Mat = np.zeros((13,13))
for key in MeasuredRelativePoses.iterkeys():
	
	Mat[transformationKey[key][1]  ,transformationKey[key][0]  ] = 1

# print Mat


def ActualPoseWrt1 (x):

	nextRow = x
	first = True
	itercount = 0

	Y = np.identity(4)
	while nextRow > 1:


		if first == True:
			j = x
			
		else:
			j = nextRow 
			
		for i in range (1,13): 

			if Mat[j,i] == 1.0:
				
				nextRow = i
				first = False 
				

				print j, "-->", i, "-->"
				RodParam = (CorrectedPose["{}".format(i)+"R" "{}".format(j)])
				stepMat = RodriguesToTransf(RodParam)

				Y = np.matmul(stepMat,Y)

				break

		itercount = itercount  +1 
		if itercount > 100:
			print "nahi hain aage ka" 
			break 
	return Y 

# for i in range (1,13):
# 	print "------"
# 	print "Pose of ", i, "wrt 1" 
# 	K = ActualPoseWrt1 (i)
# 	print K



print "..........................."
# a =  ID2toID1Estimate(TransfMatIDtoCent[transformationKey['2R8'][0,0]], TransfMatIDtoCent[transformationKey['2R8'][1,0]])
# b =  ID2toID1Estimate(TransfMatIDtoCent[transformationKey['1R2'][0,0]], TransfMatIDtoCent[transformationKey['1R2'][1,0]])


# print np.matmul(RodriguesToTransf(b),RodriguesToTransf(a)) 
print ".................."

# print np.matmul(np.linalg.inv(RodriguesToTransf(CorrectedPose["1R2"])),(RodriguesToTransf(CorrectedPose["2R8"])))

# print EstimatedPose[6] 
# print RodriguesToTransf(CorrectedPose["5R6"])