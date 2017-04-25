import rospy
from std_msgs.msg import Float64

class DepthController():
    def __init__(self):
        
        qubo_namespace = '/qubo/'

        depth = 5 #initial depth

        sensor_sub = rospy.Subscriber(qubo_namespace + 'depth', Float64, self.sensor_callback,queue_size=10)
        command_pub = rospy.Publisher(qubo_namespace + 'depth_command', Float64, queue_size=10)
        
    
    
    def update(self):
        rospy.spinonce()
        prev_time = rospy.get_time()
        pub.publish(hello_str)

    def sensor_callback(self,data):
        depth = data.data
        

if __name__ == '__main__':
    
    rospy.init_node('depth_controller', anonymous=False)
    
    dc = DepthController()
    
    while not rospy.is_shutdown():
        dc.update()
        sleep(4)
                                      
    
    
