#!/usr/bin/env python

import cv2
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from vision.srv import *
import numpy as np
import math
class vision_server:

	def __init__(self):
		self.mako_sub = rospy.Subscriber("mako_feed", Image, self.imgmsg_to_cv2_callback, "mako")
		self.hull_sub = rospy.Subscriber("hull_feed", Image, self.imgmsg_to_cv2_callback, "hull")
		self.bridge = CvBridge()
		self.mako_frame = None
		self.hull_frame = None	
		self.mako_rx = 0
		self.hull_rx = 0 

	def imgmsg_to_cv2_callback(self, data, camera):
		try:
			if camera == "mako":
				self.mako_frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
				self.mako_rx += 1
				self.gate_detection(0)
#				print("mako frame " + str(self.mako_rx) + " received")
#				cv2.imshow("mako_feed", self.mako_frame)
#				cv2.waitKey(3)
			elif camera == "hull":
				self.hull_frame = self.bridge.imgmsg_to_cv2(data, "bgr8")		
				self.hull_rx += 1
#				print("hull frame received")
		except CvBridgeError as e:
			print(e)			

	def is_vertical(self, x):
   		x1, y1, x2, y2 = x[0]
		angle = math.atan2(y2 - y1, x2 - x1) * 180.0 / np.pi
		if (angle < 115 and angle > 65) or (angle > -115 and angle < -65):
			return True
		return False

	def gate_detection(self, args):
		image = self.mako_frame.copy()
		image = image[0:image.shape[0], 120:image.shape[1]-120]
		YCB = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
		LAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
		HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#		edges = cv2.Canny(image, 50, 150)
#		lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, None, 50, 10)


		
		# image = gray
		# for x1, y1, x2, y2 in lines[0]:
		#     print x1
		#     cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

#		if lines is not None:
#			verts = [x for x in lines if self.is_vertical(x)]
#			means = [(l[0][0] + l[0][2]) / 2 for l in verts]
#			mean = lambda x: sum(x) / len(x)
#			avg = mean(means)
#			left = mean([x for x in means if x < avg])
#			right = mean([x for x in means if x >= avg])
#
#			for i in range(0, len(lines)):
#				x1, y1, x2, y2 = lines[i][0]
#
#				angle = math.atan2(y2 - y1, x2 - x1) * 180.0 / np.pi
#				if (angle < 115 and angle > 65) or (angle > -115 and angle < -65):
#					cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
#			cv2.line(image, (left, 0), (left, 1000), (0, 0, 255), 3)
#			cv2.line(image, (right, 0), (right, 1000), (0, 0, 255), 3)
#			cv2.line(image, ((left + right)/2, 0), ((left + right)/2, 1000), (0, 255, 0), 3)

#			gate_width = 3.048
#			image_width = gray.shape[1] 
#			focal_length = image_width * 1 / gate_width # Experimentally determine values for pixels (image_with) and distance
#			gate_pixel_width = right - left 
#			distance = gate_width * focal_length / gate_pixel_width
#			middle_offset = gate_width * (right + left - image_width) / 2 
#			theta = math.atan2(middle_offset, distance)
		cv2.imshow("gate_detection", HSV)
		cv2.waitKey(3)

		return -1, -1, True


def init_vision_server():
	vs = vision_server()
	rospy.init_node('vision_server')
	rospy.spin()


if __name__ == "__main__":
	init_vision_server()
