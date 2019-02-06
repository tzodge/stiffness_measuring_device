#!/usr/bin/env python
from __future__ import print_function

import roslib
# roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def callback(data):
  print('Hi')
  try:
    cv_image = CvBridge.imgmsg_to_cv2(data, "bgr8")
  except CvBridgeError as e:
    print(e)

# global cv_image
def listener():
  rospy.init_node('image_converter', anonymous=True)
  rospy.Subscriber('/camera/image_color', Image, callback)
  rospy.spin()


    # (rows,cols,channels) = self.cv_image.shape
    # if cols > 60 and rows > 60 :
    #   cv2.circle(self.cv_image, (50,50), 10, 255)

    # cv2.imshow("Image window", self.cv_image)
    # cv2.waitKey(3)
    



# def main1():
#   ic = Image_converter()
  # print(cv_image)
  # try:
  #   rospy.spin()
  # except KeyboardInterrupt:
  #   print("Shutting down")
  # image = ic.cv_image
  # cv2.imshow("ojui",image)
  # cv2.destroyAllWindows()

if __name__ == '__main__':
  listener()
