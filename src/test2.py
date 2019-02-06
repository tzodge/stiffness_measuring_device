#!/usr/bin/env python
from __future__ import print_function

import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import Markers_flipping
import numpy as np

class Image_converter:
	def __init__(self):
		self.frame = np.zeros((120,120), dtype=np.uint8)
		rospy.init_node('image_node', anonymous=True)
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber("/camera/image_color", Image, self.callback)
		self.rate = rospy.Rate(10)
		# print('Test1')

	def callback(self, data):
		# print('Test2')
		self.frame = CvBridge().imgmsg_to_cv2(data, "bgr8")
		rospy.loginfo("I heard data")

	def get_frame(self):
		img = self.frame
		print(img)
		return img

def run():
	while not rospy.is_shutdown():
		ic = Image_converter()
		img = ic.get_frame()
		print(img)
		# cv2.imshow("image",img)
		# cv2.waitKey(100)
		ic.rate.sleep()

if __name__ == '__main__':
	run()
 