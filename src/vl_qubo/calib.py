#!/usr/bin/env python

import serial
import time


ser = serial.Serial('/dev/ttyACM1',115200, timeout=0,parity=serial.PARITY_NONE,stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS)
time.sleep(3)


msg = 't,{0},{0},{0},{0},{0},{0},{0},{0}!'

while True:
    speed = input('enter speed: ')
    m = msg.format(str(speed))
    print(m) 
    ser.write(m)
