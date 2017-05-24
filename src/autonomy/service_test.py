import ram_msgs.srv
import rospy


buoy_detect = rospy.ServiceProxy('buoy_detect', ram_msgs.srv.bool_bool)
print "here"

print buoy_detect(1)
