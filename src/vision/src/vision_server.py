#!/usr/bin/env python

import cv2
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from vision.srv import *

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
#				print("mako frame " + str(self.mako_rx) + " received")
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

	def gate_detection(self, args);
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(gray, 50, 150)
		lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, None, 50, 10)
		
		# image = gray
		# for x1, y1, x2, y2 in lines[0]:
		#     print x1
		#     cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

		verts = [x for x in lines if is_vertical(x)]
		means = [(l[0][0] + l[0][2]) / 2 for l in verts]

		mean = lambda x: sum(x) / len(x)
		avg = mean(means)
		left = mean([x for x in means if x < avg])
		right = mean([x for x in means if x >= avg])
		print means
		if lines is not None:
			for i in range(0, len(lines)):
				x1, y1, x2, y2 = lines[i][0]

				angle = math.atan2(y2 - y1, x2 - x1) * 180.0 / np.pi
				if (angle < 115 and angle > 65) or (angle > -115 and angle < -65):

		focal_length = 1; # experimentally determine this
		image_width = 
		
		distance = right - left 


def init_vision_server():
	vs = vision_server()
	rospy.init_node('vision_server')
	rospy.spin()


if __name__ == "__main__":
	init_vision_server()
