#!/usr/bin/env python

import serial, time, sys, select

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


#!!! this also restarts the arduino! (apparently)
ser = serial.Serial('/dev/cu.usbmodem1421',115200, timeout=0,parity=serial.PARITY_NONE,stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS)
time.sleep(3)

print("setting low throttle")



# cmd = make_cmd_str([200] * num_thrusters)
# print cmd
# ser.write(cmd)
# time.sleep(10)

while(True):
    i = input('enter a number to try')
    cmd = make_cmd_str([i] * num_thrusters)
    print cmd
    ser.write(cmd)
    

    # for i in range(0,4095, 100):
    #     cmd = make_cmd_str([i] * num_thrusters)
    #     print cmd
    #     ser.write(cmd)
    #     time.sleep(3)
    
    # cmd = make_cmd_str([4000] * num_thrusters)
    # ser.write(cmd)
    #TODO make sure the arduino is listening!
    #TODO get this to write to the TVA too
    
while(True):
    resp = ser.readline()
    if resp:
        print resp
        if resp == "PCA initialized\n":
            print "let's go"
            break
        


