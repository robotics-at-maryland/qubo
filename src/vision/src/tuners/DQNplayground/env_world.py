#!/usr/bin/env python

from gym_env import gym_env_world
import rospy

gym_env = gym_env_world('world')
rospy.spin()
