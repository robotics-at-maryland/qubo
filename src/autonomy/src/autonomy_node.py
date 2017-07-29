#! /usr/bin/env python

import roslib; roslib.load_manifest('vision')
import rospy
from std_msgs.msg import Float64

# Brings in the SimpleActionClient
import actionlib

# Brings in the messages used by the vision action, including the
# goal message and the result message.
import ram_msgs.msg


def roll_callback(msg):
    #rospy.loginfo(rospy.get_caller_id() + "I heard %s", msg.data)
    roll = msg.data

def pitch_callback(msg):
    #rospy.loginfo(rospy.get_caller_id() + "I heard %s", msg.data)
    pitch = msg.data

def yaw_callback(msg):
    #rospy.loginfo(rospy.get_caller_id() + "I heard %s", msg.data)
    yaw = msg.data

def depth_callback(msg):
    #rospy.loginfo(rospy.get_caller_id() + "I heard %s", msg.data)
    depth = msg.data

def surge_callback(msg):
    #rospy.loginfo(rospy.get_caller_id() + "I heard %s", msg.data)
    surge = msg.data

def sway_callback(msg):
    #rospy.loginfo(rospy.get_caller_id() + "I heard %s", msg.data)
    sway = msg.data


#we need to pass this along, may need to scale it here but probably better done on the controls side
def feedback_callback(feedback):
    print feedback
    yaw_pub.pub(yaw + feedback)



def call_action(action_name):
    # Creates the SimpleActionClient, passing the type of the action
    # (VisionExampleAction) to the constructor.

    print "0"

    client = actionlib.SimpleActionClient(action_name, ram_msgs.msg.VisionNavAction)
    print "1"
    # Waits until the action server has started up and started
    # listening for goals.
    print client.wait_for_server()
    print "2"
    # Creates a goal to send to the action server.
    goal = ram_msgs.msg.VisionNavGoal(test_goal = False)
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

    rospy.init_node('autonomy_node')

    qubo_namespace = '/qubo/'
    
    #we publish commands
    roll_pub  = rospy.Publisher(qubo_namespace + "roll_cmd"  , Float64, queue_size = 10)
    pitch_pub = rospy.Publisher(qubo_namespace + "pitch_cmd" , Float64, queue_size = 10)
    yaw_pub   = rospy.Publisher(qubo_namespace + "yaw_cmd"   , Float64, queue_size = 10)
    depth_pub = rospy.Publisher(qubo_namespace + "depth_cmd" , Float64, queue_size = 10)
    surge_pub = rospy.Publisher(qubo_namespace + "surge_cmd" , Float64, queue_size = 10)
    sway_pub  = rospy.Publisher(qubo_namespace + "sway_cmd"  , Float64, queue_size = 10)

    # we subscribe to measurements
    rospy.Subscriber(qubo_namespace + "roll"  , Float64, roll_callback )
    rospy.Subscriber(qubo_namespace + "pitch" , Float64, pitch_callback)
    rospy.Subscriber(qubo_namespace + "yaw"   , Float64, yaw_callback  )
    rospy.Subscriber(qubo_namespace + "depth" , Float64, depth_callback)
    rospy.Subscriber(qubo_namespace + "surge" , Float64, surge_callback)
    rospy.Subscriber(qubo_namespace + "sway"  , Float64, sway_callback  )

    # go straight..

    surge_pub.publish(50)

    call_action('gate_action')
    
    rospy.spin()

#    result = vision_client()
