import pygame
import time

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
size = [500, 700]
screen = pygame.display.set_mode(size)

pygame.display.set_caption("My Game")

#Loop until the user clicks the close button.
done = False

delay = 100
interval = 50
clock = pygame.time.Clock()
#pygame.key.set_repeat(delay, interval)

pygame.joystick.init()

#really this should be passed in or something but for now if you want to change the name just do it here
robot_namespace = "/qubo/"
effort = 20

num_thrusters = 8

#rospy spins all these up in their own thread, no need to call spin()

#thruster_pub = rospy.Publisher(robot_namespace + "thruster_cmds"  , Float64MultiArray, queue_size = 10)

# thruster_msg = Float64MultiArray()

#ygame.key.set_repeat(10,10)

while done==False:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            
    joystick = pygame.joystick.Joystick(0);
    joystick.init()
    
    
    backwards = joystick.get_axis( 1 );
    print("backwards: " + str(backwards))
    right = joystick.get_axis(0);
    clockwise = joystick.get_axis(2);
    up = joystick.get_button(0);
    down = joystick.get_button(1);

    print("backwards: " + str(backwards) + ", right: " + str(right) + ", clockwise: " + str(clockwise) + ", up: " + str(up) + ", down: " + str(down))
            
    clock.tick(20)

pygame.quit()
