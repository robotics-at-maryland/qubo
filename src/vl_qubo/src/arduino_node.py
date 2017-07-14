#!/usr/bin/env python

import serial, time, sys, select
import rospy

num_thrusters = 8




#reads a command from stdin
def read_cmd_stdin():
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
      line = sys.stdin.readline()

      line = line.rstrip()

      if line:
        num_bytes = ser.write(line)
        print  "bytes sent =", num_bytes  

        
def make_cmd_str(thruster_cmds):
    cmd_str = "t"
    for cmd in thruster_cmds:
        cmd_str += (",")
        cmd_str += (str(cmd))

    cmd_str += ("!")
    return cmd_str


##------------------------------------------------------------------------------
# callbacks
def roll_callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

def pitch_callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

def yaw_callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

def depth_callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

def surge_callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

def sway_callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)


##------------------------------------------------------------------------------
# main
if __name__ == '__main__':


    
    #!!! this also restarts the arduino! (apparently)
    ser = serial.Serial('/dev/cu.usbmodem1421',115200, timeout=0,parity=serial.PARITY_NONE,stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS)
    time.sleep(3)
    
    
    qubo_namespace = "/qubo/"
    
    
    rospy.init_node('arduino_node', anonymous=False)

    depth_pub = rospy.Publisher(qubo_namespace + 'depth', Int, queue_size = 10)
    
    rospy.Subscriber(qubo_namespace + "roll_cmd"  , Int, roll_callback)
    rospy.Subscriber(qubo_namespace + "pitch_cmd" , Int, pitch_callback)
    rospy.Subscriber(qubo_namespace + "yaw_cmd"   , Int, yaw_callback)
    rospy.Subscriber(qubo_namespace + "depth_cmd" , Int, depth_callback)
    rospy.Subscriber(qubo_namespace + "surge_cmd" , Int, surge_callback)
    rospy.Subscriber(qubo_namespace + "sway_cmd"  , Int, sway_callback)

