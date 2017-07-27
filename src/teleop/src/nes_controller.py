import pygame
import rospy

from std_msgs.msg import Float64

#pygame setup
pygame.init()
pygame.display.set_mode([100,100])

delay = 100
interval = 50
pygame.key.set_repeat(delay, interval)

#really this should be passed in or something but for now if you want to change the name just do it here
robot_namespace = "qubo/"
thrust_effort = 128


rospy.init_node('nes_node', anonymous=False)

#rospy spins all these up in their own thread, no need to call spin()
roll_pub  =  rospy.Publisher(robot_namespace + "roll_cmd"  , Float64, queue_size = 10 )
pitch_pub =  rospy.Publisher(robot_namespace + "pitch_cmd" , Float64, queue_size = 10 )
yaw_pub   =  rospy.Publisher(robot_namespace + "yaw_cmd"   , Float64, queue_size = 10 )
depth_pub =  rospy.Publisher(robot_namespace + "depth_cmd" , Float64, queue_size = 10 )
surge_pub =  rospy.Publisher(robot_namespace + "surge_cmd" , Float64, queue_size = 10 )
sway_pub  =  rospy.Publisher(robot_namespace + "sway_cmd"  , Float64, queue_size = 10 )


pygame.joystick.init()
print pygame.joystick.get_count()
joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]

print joysticks[0].get_name()
joysticks[0].init()

joystick_count = pygame.joystick.get_count()
print joystick_count

#NES style controller map
# X = 0
# A = 1
# B = 2
# Y = 3
# L = 4
# R = 5
# Select = 6
# Start = 7

while(True):
    for event in pygame.event.get():
        if event.type == pygame.JOYBUTTONDOWN:

            # yaw buttons, Y turns counter clockwise , A turns clockwise
            if event.button == 1:
                yaw_pub.publish(thrust_effort)
            elif event.button == 3:
                yaw_pub.publish(-thrust_effort)

            # depth buttons L goes down R goes up
            elif event.button == 4:
                depth_pub.publish(thrust_effort)
            elif event.button == 5:
                depth_pub.publish(-thrust_effort)

        elif event.type == pygame.JOYAXISMOTION:
            # the game pad is used for translation IE surge and sway
            sway_pub.publish(-joysticks[event.joy].get_axis(1))
            surge_pub.publish(-joysticks[event.joy].get_axis(0))
            
            
