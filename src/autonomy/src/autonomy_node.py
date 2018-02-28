#! /usr/bin/env python

import roslib; roslib.load_manifest('vision')
import rospy
from std_msgs.msg import Float64

# Brings in the SimpleActionClient
import actionlib

# Brings in the messages used by the vision action, including the
# goal message and the result message.
import ram_msgs.msg
import ram_msgs.srv

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

    # client = actionlib.SimpleActionClient(action_name, ram_msgs.msg.VisionExampleAction)
    print "1"
    # Waits until the action server has started up and started
    # listening for goals.
    # print client.wait_for_server()
    print "2"
    # Creates a goal to send to the action server.
    # goal = ram_msgs.srv.VisionNav(test_goal = False)
    print "3"
    # Sends the goal to the action server.
    # client.send_goal(goal, feedback_cb = feedback_callback)
    print "4"
    # Waits for the server to finish performing the action.
    # client.wait_for_result()
    print "5"
    # Prints out the result of executing the action
    # return client.get_result()

if __name__ == '__main__':
    
    rospy.init_node('autonomy_node')

    # rospy.wait_for_service('depth_toggle')
    toggle_test = rospy.ServiceProxy('depth_toggle', ram_msgs.srv.bool_bool)

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

    roll_target = rospy.Publisher(qubo_namespace + "roll_target"  , Float64, queue_size = 10)
    pitch_target = rospy.Publisher(qubo_namespace + "pitch_target" , Float64, queue_size = 10)
    yaw_target = rospy.Publisher(qubo_namespace + "yaw_target"   , Float64, queue_size = 10)
    depth_target = rospy.Publisher(qubo_namespace + "depth_target" , Float64, queue_size = 10)
    surge_target = rospy.Publisher(qubo_namespace + "surge_target" , Float64, queue_size = 10)
    sway_target = rospy.Publisher(qubo_namespace + "sway_target"  , Float64, queue_size = 10)

    # go straight..

    roll = 0
    pitch = 0
    yaw = 0
    surge = 0
    sway = 0
    depth = 0
    
    roll_hold = roll
    pitch_hold = pitch
    yaw_hold = yaw
    surge_hold = surge
    sway_hold = sway
    depth_hold = depth

    rospy.sleep(10.)

    #roll_target.publish(roll_hold)
    #pitch_target.publish(pitch_hold)
    #yaw_target.publish(yaw_hold)
    #surge_target.publish(surge_hold)
    #sway_target.publish(sway_hold)
    # depth_target.publish(depth_hold)

    #_pub commands output constant thrust in arbitrary units
    #_target commands attempt to match target depth or angle

    #depth_target: + is down, , - is go up (not sure of units)
    #pitch_target: + is pitch-up moment in radians, 0 is neutral position [pi,pi]
    #roll_target: + is roll with right side dipping (maybe - not sure), 0 is neutral position [-pi,pi]
    #yaw_target: + is turn left, 0 is neutral position [-pi,pi]
    #surge_target: + is move backwards, not sure of units because robot never stops and also doesn't move in a straight line
    #sway_target: + is move to the right, not sure of units because robot never stops 

    toggle_test(0)
    depth_pub.publish(10)
    
    call_action('gate_action')
    
    rospy.spin()

#    result = vision_client()
