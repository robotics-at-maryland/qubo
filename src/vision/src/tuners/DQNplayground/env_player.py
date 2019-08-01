#!/usr/bin/env python
from __future__ import print_function
import gym_env
import rospy
#from get_keys import key_check keys_to_output
import models
import gym
from DQN import *
import itertools

print("For testing, Using OpenAi gym as warming up environment")

use_cuda = torch.cuda.is_available()

gym_env = gym_env.gym_env_you('play1')
arch = 'Qestimator_resnet18'
model = models.__dict__[arch]( num_label = 2 )
optimizer = optim.RMSprop( model.parameters() )

    # this two line is equal to model = models.Qestimator_resnet18(num_label)
if use_cuda : model.cuda()
#r_memory= ReplayMemory( 3000, BATCH_SIZE )
resize = T.Compose([T.ToPILImage(),
                        T.Scale(256, interpolation=Image.CUBIC),
                        T.ToTensor()])
if os.path.exists( os.path.join(pwd,'checkpoint/model_best.pth.tar' ) ):
         load_checkpoint( model, optimizer, arch = arch )

print('reset successful? ', gym_env.reset_env() )
rate = rospy.Rate(10)
while not rospy.is_shutdown():

    for t in itertools.count():

        #env = gym.make('CartPole-v0').unwrapped #dummy env
        state = gym_env.observe_state()

        #keys = key_check()
        #action = keys_to_output(keys)
        #action = random.randrange(2)
        #action = env.action_space.sample()

        action = select_action_model( state2tensor(state , resize), model )
        action = action[0,0]
        if not action == None:
            r,done = gym_env.apply_action( int(action) )
        rate.sleep()
        if done:
            gym_env.reset_env()
            print("complete one cycle with durations_t: {}".format(t) )
