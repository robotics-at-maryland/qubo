#!/usr/bin/env python

#sgillen - this program serves as a node that offers the arduino up to the rest of the ros system.


# the packets send to the arduino should be in the following format: p<data>!
# p tells the arduino which command to execute, the data that follows will depend on which command this is
# in general the , character is used as a delimiter, and the ! is used to mark the end of the message

# commands so far
# t,x0,x1,x2,x3,x4,x5,x6,x7!    - this sets all 8 thruster values
# d!                            - this requests the most recent depth value from the arduino (TODO)

import serial, time, sys, select
import rospy
from std_msgs.msg import Int64, Float64, String, Float64MultiArray
from std_srvs.srv import Empty, EmptyResponse

THRUSTER_INVALID = '65535'
STATUS_OK = '0'
STATUS_TIMEOUT = '1'
STATUS_OVERHEAT = '2'
STATUS_OVERHEAT_WARNING = '3'

V_START = 2.0

device = '/dev/arduino'

# When this gets flipped, send shutdown signal
shutdown_flag = False

control_domain = (-128.0, 128.0)
arduino_domain = (1029.0, 1541.0)
num_thrusters = 8

##command variables (should we make this module a class??)
roll_cmd  = 0
pitch_cmd = 0
yaw_cmd   = 0
depth_cmd = 0
surge_cmd = 0
sway_cmd  = 0

# Maps values from control_domain to arduino_domain
def thruster_map(control_in):
    ratio = (arduino_domain[1] - arduino_domain[0]) / (control_domain[1] - control_domain[0])
    return int(round((control_in - control_domain[0]) * ratio + arduino_domain[0]))

#reads a command from stdin
def read_cmd_stdin():
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
      line = sys.stdin.readline()
      line = line.rstrip()
      if line:
        num_bytes = ser.write(line)
        print  "bytes sent =", num_bytes


#sends an array of ints to the thrusters using the agreed upon protocol
#the actual over the wire value is t,x1,x2,x3,x4,x5,x6,x7,x8!
def send_thruster_cmds(thruster_cmds):
    cmd_str = "t"
    for cmd in thruster_cmds:
        cmd_str += (",")
        cmd_str += (str(cmd))

    cmd_str += ("!")
    ser.write(cmd_str)
    #print "arduino return", ser.readline()
    ##TODO parse return value

# requests depth from arduino, and waits until it receives it
def get_depth():
    ser.write('d!')
    # blocks forever until receives a newline
    depth = ser.readline()
    return depth


##------------------------------------------------------------------------------
# callbacks
def shutdown_thrusters(srv):
    global shutdown_flag
    shutdown_flag = True
    return EmptyResponse()

def roll_callback(msg):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", msg.data)
    roll_cmd = thruster_map(msg.data)

def pitch_callback(msg):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", msg.data)
    pitch_cmd = thruster_map(msg.data)

def yaw_callback(msg):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", msg.data)
    yaw_cmd = thruster_map(msg.data)

def depth_callback(msg):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", msg.data)
    depth_cmd = thruster_map(msg.data)

def surge_callback(msg):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", msg.data)
    surge_cmd = thruster_map(msg.data)

def sway_callback(msg):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", msg.data)
    sway_cmd = thruster_map(msg.data)

def thruster_callback(msg):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", msg.data)
    for i in range(0,num_thrusters):
        thruster_cmds[i] = thruster_map(msg.data[i])
        #print "after map -" , thruster_cmds[i]

##------------------------------------------------------------------------------
# main
if __name__ == '__main__':
    #!!! this also restarts the arduino! (apparently)

    # Keep trying to open serial
    while True:
        try:
            ser = serial.Serial(device,115200, timeout=0,parity=serial.PARITY_NONE,stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS)
            break
        except:
            time.sleep(0.25)
            continue

    time.sleep(3)


    #I can't think of a situation where we want to change the namespace but I guess you never know
    qubo_namespace = "/qubo/"

    rospy.init_node('arduino_node', anonymous=False)

    status_pub = rospy.Publisher(qubo_namespace + 'status', String, queue_size = 10)
    depth_pub = rospy.Publisher(qubo_namespace + "depth", Float64, queue_size = 10)

    thruster_sub = rospy.Subscriber(qubo_namespace + "thruster_cmds", Float64MultiArray, thruster_callback)

    #rospy spins all these up in their own thread, no need to call spin()
    rospy.Subscriber(qubo_namespace + "roll_cmd"  , Float64, roll_callback)
    rospy.Subscriber(qubo_namespace + "pitch_cmd" , Float64, pitch_callback)
    rospy.Subscriber(qubo_namespace + "yaw_cmd"   , Float64, yaw_callback)
    rospy.Subscriber(qubo_namespace + "depth_cmd" , Float64, depth_callback)
    rospy.Subscriber(qubo_namespace + "surge_cmd" , Float64, surge_callback)
    rospy.Subscriber(qubo_namespace + "sway_cmd"  , Float64, sway_callback)

    rospy.Service(qubo_namespace + "shutdown_thrusters", Empty, shutdown_thrusters)

    thruster_cmds = [thruster_map(0)]*num_thrusters

    rate = rospy.Rate(10) #100Hz

    # Poll the ina for voltage, start up regular
    startup_voltage = 0.0

    # zero the thrusters
    send_thruster_cmds([0] * num_thrusters)
    zero = ser.readline().strip()

    while startup_voltage <= V_START:
        ser.write('s!')
        startup_voltage = float(ser.readline())
        time.sleep(0.1)


    while not rospy.is_shutdown():

        depth = get_depth() #TODO
        depth_pub.publish(depth)


        #thruster layout found here https://docs.google.com/presentation/d/1mApi5nQUcGGsAsevM-5AlKPS6-FG0kfG9tn8nH2BauY/edit#slide=id.g1d529f9e65_0_3

        #surge, yaw, sway thrusters

        # thruster_cmds[0] += (surge_cmd - yaw_cmd - sway_cmd)
        # thruster_cmds[1] += (surge_cmd + yaw_cmd + sway_cmd)
        # thruster_cmds[2] += (surge_cmd + yaw_cmd - sway_cmd)
        # thruster_cmds[3] += (surge_cmd - yaw_cmd + sway_cmd)

        # #depth, pitch, roll thrusters
        # thruster_cmds[4] += (depth_cmd + pitch_cmd + roll_cmd)
        # thruster_cmds[5] += (depth_cmd + pitch_cmd - roll_cmd)
        # thruster_cmds[6] += (depth_cmd - pitch_cmd - roll_cmd)
        # thruster_cmds[7] += (depth_cmd - pitch_cmd + roll_cmd)


        # Build the thruster message to send
        if shutdown_flag:
            send_thruster_cmds([0] * num_thrusters)
        else:
            send_thruster_cmds(thruster_cmds)
        # print "hello"

        #ser.write('c!')
        #temp = ser.readline()
        #print(temp)

        # Get the status
        thrust = ser.readline().strip()
        status = ser.readline().strip()

        print(thruster_cmds)
        print(thrust, status)

        if thrust == THRUSTER_INVALID:
            print('Invalid thruster input')

        if status == STATUS_OK:
            print('STATUS OK')
            status_pub.publish(data='OK')
        elif status == STATUS_TIMEOUT:
            print('STATUS TIMEOUT')
            status_pub.publish(data='TIMEOUT')
        elif status == STATUS_OVERHEAT:
            print('STATUS OVERHEAT')
            status_pub.publish(data='OVERHEAT')
        elif status == STATUS_OVERHEAT_WARNING:
            print('STATUS OVERHEAT WARNING')
            status_pub.publish(data='OVERHEAT WARNING')

        rate.sleep()
