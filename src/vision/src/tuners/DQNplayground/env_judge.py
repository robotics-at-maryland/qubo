#!/usr/bin/env python
from __future__ import print_function
import gym_env
import rospy
import cv2
 #!/usr/bin/env python
import rospy
#from std_msgs.msg import UInt8MultiArray

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String
bridge = CvBridge()

def callback(data):
    cv2.imshow( "window",bridge.imgmsg_to_cv2( data, desired_encoding="passthrough" ) )
    if cv2.waitKey(1) and False: pass
    rospy.loginfo( rospy.get_caller_id() + "I see it" )

def monitor():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("world_vision", Image, callback )
    rospy.spin()

if __name__ == '__main__':
    monitor()
