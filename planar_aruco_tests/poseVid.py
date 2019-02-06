import numpy as np 
import cv2
# from Ipython import embed


cap = cv2.VideoCapture(0)

#type (cap)
print type (cap)
while (True):
	ret, frame = cap.read()
	# print type (ret)
	# print type (frame)

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
	#print type(corners) 
	# print " gap \n"
	if type (corners) == np.ndarray:
		print(corners)
        
		# embed()
		# firstCorner = tuple(map(tuple,corners[2*3,:]))	
		# fifthCorner = tuple(map(tuple,corners[2*5,:]))
		# # firstCorner = np.int32(firstCorner)
		# fifthCorner = np.int32(fifthCorner) 
		
		firstCorner = corners[2*3,:]	
		fifthCorner = corners[2*5,:]
		# print(firstCorner[0])
		# print(fifthCorner)
		# break


		cv2.line(gray,tuple(firstCorner[0]),tuple(fifthCorner[0]),(255),5)
		# print  fifthCorner 
		#cv2.line(gray,(0,0),(511,511),(255,0,0),5)
		print "aa gaya"




	cv2.imshow('frame',gray)

	
	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break


cap.release()
cv2.destroyAllWindows()




#------------------------------------------
# import numpy as np
# import cv2

# cap = cv2.VideoCapture(0)

# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Display the resulting frame
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
