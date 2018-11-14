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

def move_start(speed, angle):
    #hold these orientations constant
    roll_target.publish(roll)
    pitch_target.publish(pitch)

    #hold depth constant
    depth_target.publish(depth)
    
    #there's no sway controller anyways
    #sway_target.publish(sway)

    #assuming angle is in radians and positive is CCW
    yaw_target.publish(yaw+angle)
    
    rospy.sleep(2.)

    #not sure what to do about these units
    surge_target.publish(speed)

def move_end():
    surge_target.publish(0)

#careful using high values for speed, the robot has no way to stop    
def move_time(speed, time, angle):
    if time == 0:
        return
    
    #hold these orientations constant
    roll_target.publish(roll)
    pitch_target.publish(pitch)

    #hold depth constant
    depth_target.publish(depth)

    #assuming angle is in radians and positive is CCW
    yaw_target.publish(yaw+angle)

    if angle != 0:
        rospy.sleep(2.)

    surge_target.publish(speed)
    rospy.sleep(time)
    surge_target.publish(0)

if __name__ == '__main__':
    
    rospy.init_node('autonomy_node')

    # rospy.wait_for_service('depth_toggle')
    toggle_test = rospy.ServiceProxy('depth_toggle', ram_msgs.srv.bool_bool)
    pos_vel_test = rospy.ServiceProxy('toggle_pos_vel', ram_msgs.srv.bool_bool)

    qubo_namespace = '/qubo/'

    #we publish commands
    roll_pub  = rospy.Publisher(qubo_namespace + "roll_cmd"  , Float64, queue_size = 10)
    pitch_pub = rospy.Publisher(qubo_namespace + "pitch_cmd" , Float64, queue_size = 10)
    yaw_pub   = rospy.Publisher(qubo_namespace + "yaw_cmd"   , Float64, queue_size = 10)
    depth_pub = rospy.Publisher(qubo_namespace + "depth_cmd" , Float64, queue_size = 10)
    surge_pub = rospy.Publisher(qubo_namespace + "surge_cmd" , Float64, queue_size = 10)
    sway_pub  = rospy.Publisher(qubo_namespace + "sway_cmd"  , Float64, queue_size = 10)

    # we subscribe to measurements
    roll_sub = rospy.Subscriber(qubo_namespace + "roll"  , Float64, roll_callback )
    pitch_sub = rospy.Subscriber(qubo_namespace + "pitch" , Float64, pitch_callback)
    yaw_sub = rospy.Subscriber(qubo_namespace + "yaw"   , Float64, yaw_callback  )
    depth_sub = rospy.Subscriber(qubo_namespace + "depth" , Float64, depth_callback)
    surge_sub = rospy.Subscriber(qubo_namespace + "surge" , Float64, surge_callback)
    sway_sub = rospy.Subscriber(qubo_namespace + "sway"  , Float64, sway_callback  )

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

    #to control surge and sway with velocity (default), use pos_vel_test(0)
    #to control surge and sway with position, use pos_vel_test(1)
    
    rospy.sleep(1.)
    print("switching to position control")
    pos_vel_test(1)
    
    #to turn a controller off, create a ServiceProxy line as shown above, and then turn off using service_name(0) and on using service_name(1)
    #the controller must be turned off for *_publish commands to work
    
    rospy.spin()
