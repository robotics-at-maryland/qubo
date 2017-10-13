#!/usr/bin/env python

from __future__ import print_function
import rospy
import gym
from std_msgs.msg import String, Float32, UInt8MultiArray, Bool
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from vision.srv import *
import cv2
import numpy as np
import thread
#GYM_Apply_Action, GYM_Req_Observe, GYM_Req_Reset

"""
'Bool', 'Byte', 'ByteMultiArray', 'Char', 'ColorRGBA',
'Duration', 'Empty', 'Float32', 'Float32MultiArray',
'Float64', 'Float64MultiArray', 'Header', 'Int16',
'Int16MultiArray', 'Int32', 'Int32MultiArray', 'Int64'
, 'Int64MultiArray', 'Int8', 'Int8MultiArray',
'MultiArrayDimension', 'MultiArrayLayout', 'String',
'Time', 'UInt16', 'UInt16MultiArray',
 'UInt32', 'UInt32MultiArray', 'UInt64',
  'UInt64MultiArray', 'UInt8', 'UInt8MultiArray'
"""

# Game play component
# https://github.com/Sentdex/pygta5/blob/
# Citation: Box Of Hats (https://github.com/Box-Of-Hats )

#import win32api as wapi
#import time

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'$/\\":
    keyList.append(char)

class gym_env_you(object):
    def __init__(self,name=None):
        self.name = name
        #self.transforms = transforms
        rospy.wait_for_service('req_screen')
        self.observe_screen_req = rospy.ServiceProxy('req_screen', GYM_Req_Observe )#AddTwoInts)
        rospy.wait_for_service('apply_action')
        self.apply_action_req = rospy.ServiceProxy('apply_action', GYM_Apply_Action )#AddTwoInts)
        rospy.wait_for_service('reset_env')
        self.reset_req = rospy.ServiceProxy('reset_env', GYM_Req_Reset )#AddTwoInts)
        self.last_screen = 0
        self.screen_width = 600 # this is based on the code from gym

        self.name = name if not name == None else 'you'
        rospy.init_node(self.name,anonymous=True)

        self.bridge = CvBridge()

    def observe_state(self,**args):

        try:
            resq1 = self.observe_screen_req(True)
            try:
                screen = self.bridge.imgmsg_to_cv2(resq1.IMAGE, desired_encoding="passthrough")
            except CvBridgeError as e:
                print(e)

            cart_location = resq1.location

            screen = screen[:, 160:320]
            view_width = 320

            if cart_location < view_width // 2:
                slice_range = slice(view_width)
            elif cart_location > (self.screen_width - view_width // 2):
                slice_range = slice(-view_width, None)
            else:
                slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
                            # Strip off the edges, so that we have a square image centered on a cart
            screen = screen[:, :, slice_range]
            # Convert to float, rescare, convert to torch tensor
            # (this doesn't require a copy)
            screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
            state = screen - self.last_screen
            self.last_screen = screen

            return state

        except rospy.ServiceException, e:
            print( "Service call failed: %s"%e )

    def apply_action( self,action ):
        try:
            resq1 = self.apply_action_req( action )
            return resq1.reward, resq1.done

        except rospy.ServiceException, e:
            print( "Service call failed: %s"%e )

    def reset_env(self):
        try:
            resq1 = self.reset_req(True)
            return resq1.confirmation

        except rospy.ServiceException, e:
            print( "Service call failed: %s"%e )

class gym_env_world(object):
    def __init__(self,name=None):
        self.env = gym.make('CartPole-v0').unwrapped
        self.env.reset()
        # publish the world to the topics
        self.vision_pub = rospy.Publisher( 'world_vision', Image, queue_size=10)
        self.reward_pub = rospy.Publisher( 'world_reward', Float32, queue_size=10)

        # communicate with client
        self.action_service = rospy.Service( 'apply_action', GYM_Apply_Action, self.apply_action )
        self.observe_service = rospy.Service( 'req_screen',GYM_Req_Observe,self.observe_screen )
        #self.action_sub = rospy.Subscriber("apply_action", Float32, self.apply_action() )
        self.reset_service = rospy.Service( 'reset_env', GYM_Req_Reset, self.reset_env )
        #s = rospy.Service('add_two_ints', AddTwoInts, handle_add_two_ints)

        self.name = name if not name == None else 'world'
        rospy.init_node(self.name,anonymous=True)
        self.screen_width = 600

        self.bridge = CvBridge()
        self.screen = None
        self.stop_rendering = False

        thread.start_new_thread( self.self_rendering, ('world', ) )

    def self_rendering(self,threadingname):
        while not self.stop_rendering:
            self.screen = self.env.render(mode='rgb_array')#.transpose( (2, 0, 1) )

    def reset_env(self,req):
        if req.verify is True:
             self.env.reset()
             return GYM_Req_ResetResponse(True)
        else:
            return GYM_Req_ResetResponse(False)

    def get_cart_location(self):
        world_width = self.env.x_threshold * 2
        scale = self.screen_width / world_width
        return int(self.env.state[0] * scale + self.screen_width / 2.0)  # MIDDLE OF CART

    def observe_screen(self,req):

        #self.env.render(mode='rgb_array',close=True)
        screen = self.screen.copy()
        try:
            screen = self.bridge.cv2_to_imgmsg( screen, encoding="passthrough" )
        except CvBridgeError as e:
            print(e)

        print('type',type(screen))
        self.vision_pub.publish(screen)
        resp = GYM_Req_ObserveResponse()
        resp.IMAGE = screen
        resp.location = self.get_cart_location()
        return resp

    def apply_action(self,req):
        _, reward, done, _ = self.env.step( req.action ) # [0, 0])
        print("action: ", req.action)
        print( "apply_action reward", reward )
        resp = GYM_Apply_ActionResponse()
        resp.reward = reward
        resp.done = done
        return resp

    #def __del__(self):
    #    self.stop_rendering = True
    #    print('close the world')

# the key check and keys_to_output only work for windows
"""
def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys

def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
    [A,W,D] boolean values.
    '''
    #output = [0,0]
    output = None
    if 'A' in keys:
        output = 0
        #output[0] = 1
    elif 'D' in keys:
        output = 1
        #output[1] = 1
    else:
        pass
        #output[1] = 1

    return output
"""
if __name__ == "__main__":
    gym_env_you('you1')
    gym_env_world('world')
