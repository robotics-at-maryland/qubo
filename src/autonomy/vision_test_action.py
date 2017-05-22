import roslib; roslib.load_manifest('vision')
import rospy
import actionlib

from vision.msg import VisionExampleAction 
from vision.srv import bool_bool


client = actionlib.SimpleActionClient("find_buoy", ram_msgs.msg.ExampleVisionAction)
goal = ram_msgs.msg.ExampleVisionAction(test_goal = true)
client.send_goal(goal)
client.wait_for_result()
print(client.get_result())
buoy_detect(1)
