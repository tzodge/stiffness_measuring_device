import cv2 as cv
import numpy as np 
from cv2 import aruco
import glob
from scipy.optimize import minimize





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
	print x[0:3]
	
	rot,_ = cv.Rodrigues(x[0:3])
	
	
	trans =  np.reshape(x[3:6],(3,1))

	Trransf = np.concatenate((rot,trans),axis = 1)
	Trransf = np.concatenate((Trransf,np.array([[0,0,0,1]])),axis = 0)

	return Trransf




NoOfImages = 062
with np.load('B.npz') as X:
	mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
# Arrays to store object points and image points from all the images.



path = "/home/biorobotics/cpp_test/Opti/BundleAdjustmentData"
MeasuredRelativePoses = {}  #relative measured poses willbe stored here
transformationKey = {}
CorrectedPose = {}
PoseCount  = {}

for i in range (0,NoOfImages):
	fname = "BundleAdjustmentData" + "/opencv_frame_{}".format(i) + ".jpg"
	# print fname
	img = cv.imread(fname)
	aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
	parameters = aruco.DetectorParameters_create()
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	corners, ids, _ = aruco.detectMarkers(img, aruco_dict, parameters = parameters)
	if len(corners) >= 2:
		rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers( corners, 19.16, mtx,dist)
		aruco.drawDetectedMarkers(img,corners,ids)
		

		for i in range(0, len(corners)):
			for j in range (i+1, len(corners)):
				print ids[i], ids[j]

				if ids[i] > ids[j]:
					R1,_ = cv.Rodrigues(rvecs[i])
					R2,_ = cv.Rodrigues(rvecs[j])
					a = "{}".format(ids[j][0]) + "R" +"{}".format(ids[i][0]) #KeyName 
					R12 =  np.matmul(np.linalg.inv(R2),R1)
					R12,_ = cv.Rodrigues(R12)
					transl = tvecs[i] - tvecs[j]
					transl = col(transl)
					R12 = np.vstack((R12,transl))	
					# print MeasuredRelativePoses[a]
											
					if a in PoseCount:
						# print "first"
						R12 =  (R12 + PoseCount[a]*MeasuredRelativePoses[a])/(PoseCount[a] + 1) #taking average of poses between same IDS in all frames 
						PoseCount[a] = PoseCount[a] + 1 

					else:
						# print "second"
						# R12 =  np.matmul(np.linalg.inv(R2),R1)
						PoseCount[a] = 1




					MeasuredRelativePoses.update( {a : R12})
					transformationKey.update({a: np.array(([ids[j], ids[i]]))})

					



				else :
					R1,_ = cv.Rodrigues(rvecs[j])
					R2,_ = cv.Rodrigues(rvecs[i])
					a = "{}".format(ids[i][0]) + "R" +"{}".format(ids[j][0]) #KeyName 

					R12 =  np.matmul(np.linalg.inv(R2),R1)		
					R12,_ = cv.Rodrigues(R12)
					transl = tvecs[j] - tvecs[i]
					transl = col(transl)
					R12 = np.vstack((R12,transl))
					


					if a in PoseCount:
						# print "third"
						R12 =  (R12 + PoseCount[a]*MeasuredRelativePoses[a])/(PoseCount[a] + 1)
						PoseCount[a] = PoseCount[a] + 1 
					else:
						# print "fourth"
						# R12 =  np.matmul(np.linalg.inv(R2),R1)
						PoseCount[a] = 1

					MeasuredRelativePoses.update( {a : R12})
					transformationKey.update({a: np.array(([ids[i], ids[j]]))})

				print "measured"
				print MeasuredRelativePoses[a]
				print ""

# checking the images and number of detected markers  				
	# cv.imshow('image',img)
	# k = cv.waitKey(0)
	# if k%256 == 32:
	# 	break 
				




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