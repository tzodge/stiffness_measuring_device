"""
This code assumes that images used for calibration are of the same arUco marker board provided with code
"""



import cv2
from cv2 import aruco
import yaml
import numpy as np


# Set this flsg True for calibrating camera and False for validating results real time
Calibrate_camera = True


# Set number of images taken using data_generation script.
numberOfImages = 22

# Set path to the images
path = "/home/biorobotics/cpp_test/Aruco/"


# For validating results, show aruco board to camera.



aruco_dict = aruco.getPredefinedDictionary( aruco.DICT_6X6_1000 )


#Provide length of the marker's side
markerLength = 4.65  # Here, measurement unit is centimetre.

# Provide separation between markers
markerSeparation = 0.645   # Here, measurement unit is centimetre.



# create arUco board
board = aruco.GridBoard_create(4, 5, markerLength, markerSeparation, aruco_dict)


'''uncomment following block to draw and show the board'''
#img = board.draw((864,1080))
#cv2.imshow("aruco", img)




arucoParams = aruco.DetectorParameters_create()

if Calibrate_camera == True:
    
    img_list = []
    i = 0
    while i < numberOfImages:
        # print "aa gaya"
        name =  "calibData/" + str(i) + ".jpg"
        print name
        img = cv2.imread(name)
        cv2.imshow("IMAGE",img)
        img_list.append(img)
        h, w, c = img.shape
        i += 1

        
    counter = []
    corners_list = []
    id_list = []
    first = True
    for im in img_list:
    
        img_gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco_dict, parameters=arucoParams)
        #print(np.shape(corners))
        if len(ids) == 20:   
            if first == True:
                corners_list = corners
                print(np.shape(corners))
                id_list = ids
                first = False
                print("corners list size  {}".format(np.shape(corners_list)))

            else:
                #corners_list = np.hstack((corners_list, corners))
                print(np.shape(corners_list))
                print(corners)
                if(np.shape(corners_list)==np.shape(corners)):
                    id_list = np.hstack((id_list,ids))
            counter.append(len(ids))
        
    counter = np.array(counter)
    print len(id_list)
    print len(corners_list)
    print ("Calibrating camera .... Please wait...")
    #mat = np.zeros((3,3), float)
    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(corners_list, id_list, counter, board, img_gray.shape, None, None )

    print("Camera matrix is \n", mtx, "\n And is stored in calibration.yaml file along with distortion coefficients : \n", dist)
    data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
    with open("calibration.yaml", "w") as f:
        yaml.dump(data, f)
    

    
