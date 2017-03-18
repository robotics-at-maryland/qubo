#!/usr/bin/env python

import serial
import time
from curses import wrapper

ser = serial.Serial('/dev/ttyACM0',115200, timeout=0,parity=serial.PARITY_NONE,stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS)
time.sleep(3)


#set_speed(100,100)
#TODO make sure the arduino is listening!


while(True):
    #ser.write('hello!')
    str = ser.readline()
    if str:
        print str
        time.sleep(.1)
        ser.write('D')
        time.sleep(.1)
