
import rospy
import actionlib
import VisionExampleAction.msg

client = actionlib.SimpleActionClient("buoy_detect", ram_msgs.msg.ExampleVisionAction)

goal = ram_msgs.msg.ExampleVisionAction(test_goal = true)
client.send_goal(goal)
client.wait_for_result()
print(client.get_result())

buoy_detect(1)
