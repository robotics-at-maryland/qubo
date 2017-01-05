#! /usr/bin/env python

import roslib; roslib.load_manifest('vision')
import rospy

# Brings in the SimpleActionClient
import actionlib

# Brings in the messages used by the vision action, including the
# goal message and the result message.
import vision.msg

def vision_client():
    # Creates the SimpleActionClient, passing the type of the action
    # (VisionExampleAction) to the constructor.
    client = actionlib.SimpleActionClient('vision_example', vision.msg.VisionExampleAction)

    # Waits until the action server has started up and started
    # listening for goals.
    client.wait_for_server()
	
    # Creates a goal to send to the action server.
    goal = vision.msg.VisionExampleGoal(test_goal = False)

    # Sends the goal to the action server.
    client.send_goal(goal)

    # Waits for the server to finish performing the action.
    client.wait_for_result()

    # Prints out the result of executing the action
    return client.get_result()  

if __name__ == '__main__':
    try:
        # Initializes a rospy node so that the SimpleActionClient can
        # publish and subscribe over ROS.
        rospy.init_node('vision_client')
        result = vision_client()
        print result
    except rospy.ROSInterruptException:
        print "program interrupted before completion"