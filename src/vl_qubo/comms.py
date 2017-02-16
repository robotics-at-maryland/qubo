#definitions so far:

#computer sends "s,l_num,r_num!" is going to mean set left motor to l_num and right motor to righ_nu
# the arduino will send back a "A\n" to acknowledge this command

#computer sends "p!" and will get the ping sensor data in the form p,l_sense,r_sense


#In all cases the arduino will return B if it could not parse the input
import serial
import time

   
def set_speed(left, right):
    while(True):
        ser.write('p,' + str(left) + ',' + str(right) + '!')
        time.sleep(1)
        print 'waiting for P'
        ret = ser.readline().rstrip()
        print ret
        if(ret == 'P'):
            print 'P received'
            return
        


ser = serial.Serial('/dev/ttyACM0',115200, timeout=0,parity=serial.PARITY_NONE,stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS)
time.sleep(3)


#set_speed(100,100)
#TODO make sure the arduino is listening! 


while(True):
    #ser.write('hello!')
    print ser.readline()
    time.sleep(.1)
    ser.write('D')
    time.sleep(.1)
    


 
