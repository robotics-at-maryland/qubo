import rospy
from std_srvs.srv import Empty
import time

switch = False

def handler(asdf):
    switch = True 
    return Empty

rospy.init_node('test_node')
s = rospy.Service('/test', Empty, handler)

while True:
    #print(switch)
    time.sleep(1)
    continue
