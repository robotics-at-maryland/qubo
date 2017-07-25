import pygame
import rospy
import time

from std_msgs.msg import Float64

#pygame setup
pygame.init()
pygame.display.set_mode([100,100])

delay = 100
interval = 50
pygame.key.set_repeat(delay, interval)

#really this should be passed in or something but for now if you want to change the name just do it here
robot_namespace = "qubo/"
effort = 20

num_thrusters = 8


rospy.init_node('keyboard_node', anonymous=False)

#rospy spins all these up in their own thread, no need to call spin()
roll_pub  =  rospy.Publisher(robot_namespace + "roll_cmd"  , Float64, queue_size = 10 )
pitch_pub =  rospy.Publisher(robot_namespace + "pitch_cmd" , Float64, queue_size = 10 )
yaw_pub   =  rospy.Publisher(robot_namespace + "yaw_cmd"   , Float64, queue_size = 10 )
depth_pub =  rospy.Publisher(robot_namespace + "depth_cmd" , Float64, queue_size = 10 )
surge_pub =  rospy.Publisher(robot_namespace + "surge_cmd" , Float64, queue_size = 10 )
sway_pub  =  rospy.Publisher(robot_namespace + "sway_cmd"  , Float64, queue_size = 10 )

thruster_pubs = {}

for i in range(0,num_thrusters):
    thruster_pubs[i] = rospy.Publisher(robot_namespace + "thruster_cmd" +  str(i) + "_cmd", Float64, Float64, queue_size = 10)


print pygame.K_0

print pygame.K_8

pygame.key.set_repeat(10,10)

while(True):

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            print event.key

    keys_pressed = pygame.key.get_pressed()
    sway = surge = yaw = depth = 0
    
    # we really can just do = rather than += but who cares
    
    if keys_pressed[pygame.K_a]:
        sway_pub.publish(effort)

    elif keys_pressed[pygame.K_d]:
        sway_pub.publish(-effort)

    if keys_pressed[pygame.K_w]:
        surge_pub.publish(effort)
        print "asdasd"

    elif keys_pressed[pygame.K_s]:
        surge_pub.publish(-effort)

    if keys_pressed[pygame.K_q]:
        yaw_pub.publish(effort)

    elif keys_pressed[pygame.K_e]:
        yaw_pub.publish(-effort)


        
    if keys_pressed[pygame.K_r]:
        depth_pub.publish(effort)
        
    elif keys_pressed[pygame.K_f]:
        depth_pub.publish(-effort)


    #this only works because pygame.k_X is a number and k_0 - k_8 are contiguous

    if keys_pressed[pygame.K_MINUS]:
        sign = -1
    else:
        sign = 1
        
    for key in range(pygame.K_0, pygame.K_7):
        
        if keys_pressed[key]:
            thruster_pubs[i].publish[effort*sign]
        

    time.sleep(.05)
