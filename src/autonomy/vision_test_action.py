import rospy
import actionlib
import ram_msgs.msg
from std_msgs.msg import Float64

# Might make this into a class, doesn't matter too much in this case

#globals
depth = 0
yaw = -1

        
def pretend_vision_action():
    return (2 - yaw , 2 - depth)
        
#there might be some way to make these all one generic function but who cares
def depth_callback(new_depth):
    global depth
    depth = new_depth.data
    

def yaw_callback(new_yaw):
    global yaw
    yaw = new_yaw.data

    
if __name__ == '__main__':
    
    rospy.init_node('autonomy_node', anonymous=False)

    # client = actionlib.SimpleActionClient("buoy_detect", ram_msgs.msg.ExampleVisionAction)

    # goal = ram_msgs.msg.ExampleVisionAction(test_goal = true)
    # client.send_goal(goal)
    # client.wait_for_result()
    # print(client.get_result())
    
    # buoy_detect(1)

    rospy.Subscriber('/qubo/yaw', Float64, yaw_callback)
    rospy.Subscriber('/qubo/depth', Float64, depth_callback)

    yaw_target_pub = rospy.Publisher('/qubo/yaw_target', Float64, queue_size=10)
    depth_target_pub = rospy.Publisher('/qubo/depth_target', Float64, queue_size=10)
    rate = rospy.Rate(10) # 100hz

    print "hello?"
    
    while(True):
        error_x, error_y = pretend_vision_action()
        yaw_target_pub.publish(error_x + yaw)
        rate.sleep()
