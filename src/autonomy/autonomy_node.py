#! /usr/bin/env python

import roslib; roslib.load_manifest('vision')
import rospy

# Brings in the SimpleActionClient
import actionlib

# Brings in the messages used by the vision action, including the
# goal message and the result message.
import ram_msgs.msg

#we need to pass this along, may need to scale it here but probably better done on the controls side
def feedback_callback(feedback):
    print feedback

def vision_client():
    # Creates the SimpleActionClient, passing the type of the action
    # (VisionExampleAction) to the constructor.
    print "0"
    ## !!! you'll need to change the action name here to test different actions
    client = actionlib.SimpleActionClient('buoy_action', ram_msgs.msg.VisionExampleAction)
    print "1"
    # Waits until the action server has started up and started
    # listening for goals.
    print client.wait_for_server()
    print "2"
    # Creates a goal to send to the action server.
    goal = ram_msgs.msg.VisionExampleGoal(test_goal = False)
    print "3"
    # Sends the goal to the action server.
    client.send_goal(goal, feedback_cb = feedback_callback)
    print "4"
    # Waits for the server to finish performing the action.
    client.wait_for_result()
    print "5"
    # Prints out the result of executing the action
    return client.get_result()  

if __name__ == '__main__':
    
    rospy.init_node('vision_client')
    result = vision_client()
    
