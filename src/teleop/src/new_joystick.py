import pygame
import rospy
import time

from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray

# Define some colors
BLACK    = (   0,   0,   0)
WHITE    = ( 255, 255, 255)

class TextPrint:
    def __init__(self):
        self.reset()
        self.font = pygame.font.Font(None, 20)

    def print(self, screen, textString):
        textBitmap = self.font.render(textString, True, BLACK)
        screen.blit(textBitmap, [self.x, self.y])
        self.y += self.line_height
        
    def reset(self):
        self.x = 10
        self.y = 10
        self.line_height = 15
        
    def indent(self):
        self.x += 10
        
    def unindent(self):
        self.x -= 10

#pygame setup
pygame.init()
#pygame.display.set_mode([100,100])

# Set the width and height of the screen [width,height]
size = [100, 100]
screen = pygame.display.set_mode(size)

pygame.display.set_caption("If you don't draw this window it might break")

done = False

delay = 100
interval = 50
clock = pygame.time.Clock()
#pygame.key.set_repeat(delay, interval)

pygame.joystick.init()

#really this should be passed in or something but for now if you want to change the name just do it here
robot_namespace = "/qubo/"
effort = 10

num_thrusters = 8


rospy.init_node('joystick_node', anonymous=False)

#rospy spins all these up in their own thread, no need to call spin()
roll_pub  =  rospy.Publisher(robot_namespace + "roll_cmd"  , Float64, queue_size = 10 )
pitch_pub =  rospy.Publisher(robot_namespace + "pitch_cmd" , Float64, queue_size = 10 )
yaw_pub   =  rospy.Publisher(robot_namespace + "yaw_cmd"   , Float64, queue_size = 10 )
depth_pub =  rospy.Publisher(robot_namespace + "depth_cmd" , Float64, queue_size = 10 )
surge_pub =  rospy.Publisher(robot_namespace + "surge_cmd" , Float64, queue_size = 10 )
sway_pub  =  rospy.Publisher(robot_namespace + "sway_cmd"  , Float64, queue_size = 10 )


#thruster_pub = rospy.Publisher(robot_namespace + "thruster_cmds"  , Float64MultiArray, queue_size = 10)

# thruster_msg = Float64MultiArray()

#ygame.key.set_repeat(10,10)

while done==False:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
    
    joystick = pygame.joystick.Joystick(0);
    joystick.init()
    
    
    backwards = effort*joystick.get_axis( 1 );
    right = effort*joystick.get_axis(0);
    clockwise = -1*effort*joystick.get_axis(2);
    up = effort*joystick.get_button(0);
    down = effort*joystick.get_button(1);

    print("backwards: " + str(backwards) + ", right: " + str(right) + ", clockwise: " + str(clockwise) + ", up: " + str(up) + ", down: " + str(down))

    # sway = surge = yaw = depth = 0
    # thruster_msg.data = [0]*num_thrusters

    sway_pub.publish(right)
    surge_pub.publish(backwards)
    depth_pub.publish(up - down)
    yaw_pub.publish(clockwise)
        
    #this only works because pygame.k_X is a number and k_0 - k_8 are contiguous
    #for i in range(0, 8):
    #    if keys_pressed[pygame.K_0 + i]:
    #        thruster_msg.data[i] = (effort*sign)

    #print thruster_msg.data
    #thruster_pub.publish(thruster_msg)
            
    clock.tick(20)

pygame.quit()
