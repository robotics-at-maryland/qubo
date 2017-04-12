import ram_msgs.srv
import rospy


buoy_detect = rospy.ServiceProxy('service_test', ram_msgs.srv.bool_bool)
buoy_detect(1)
