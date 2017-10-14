#!/usr/bin/env python
from __future__ import print_function
import gym_env
import rospy
#from get_keys import key_check keys_to_output
import random
import gym

gym_env = gym_env.gym_env_you('play1')
print('reset successful? ', gym_env.reset_env() )
rate = rospy.Rate(10)
while not rospy.is_shutdown():
    # dummy env
    env = gym.make('CartPole-v0').unwrapped
    state = gym_env.observe_state()
    #keys = key_check()
    #action = keys_to_output(keys)
    #action = random.randrange(2)
    action = env.action_space.sample()
    if not action == None:  r,done = gym_env.apply_action( int(action) )
    if done: gym_env.reset_env()
    print("complete one cycle")
    rate.sleep()
