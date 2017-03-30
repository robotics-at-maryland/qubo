#!/usr/bin/env python

import serial, time, sys, select

ser = serial.Serial('/dev/ttyACM0',115200, timeout=0,parity=serial.PARITY_NONE,stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS)
time.sleep(3)

def input_t(str):
    str = raw_input()

#TODO make sure the arduino is listening!
#TODO get this to write to the tiva too

def input():
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
      line = sys.stdin.readline()
      if line:
        ser.write(line)

while(True):
    #ser.write('hello!')
    time.sleep(.1)
    str = ser.readline()
    if str:
        print str,
    input()
