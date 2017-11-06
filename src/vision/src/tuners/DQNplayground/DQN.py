#!/usr/bin/env python

from collections import namedtuple

from itertools import count
from copy import deepcopy
from PIL import Image
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

import pickle # dumping python data
import shutil # high level file operation

import warnings
import os

pwd = os.path.dirname(__file__)

use_cuda = torch.cuda.is_available()
#if use_cuda is False: raise Exception('Check your cuda!')

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

#Training
BATCH_SIZE = 64
GAMMA = 0.999

#Epsilon Greedy
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

steps_done = 0
last_sync = 0


Transition = namedtuple(typename="Transition",field_names=("state","action","next_state","reward") )
#API: collections.namedtuple(typename, field_names[, verbose=False][, rename=False])

class ReplayMemory(object):

    def __init__(self,capacity,batch_size=1):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.batch_size = batch_size

    def push(self,**args2):
    #def push(self,use_tuple=False,ntuple=None,*args1,**args2):
        if len(self.memory)< self.capacity:
            self.memory.append(None)
            self.memory[self.position] = Transition(**args2)
            self.position +=1
            #self.memory[self.position] = Transition(*args1,**args2)


    def sample(self,batch_size=None):
        batch_size = batch_size if batch_size!=None else self.batch_size
        return random.sample(self.memory, batch_size)

    def dump(self,path):
        with open(path,'wb'):
            pickle.dump(self.memory,path,-1) # use highest protocal

    def load(self,path):
        with open(path,'rb'):
            self.memory = pickle.load(path)

    def __len__(self):
        return len( self.memory )

def plot_durations(model,episode_durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

def select_action_model(state,model):
    return model(
        Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)

def select_action(state,model):

    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        return model(
            Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(2)]])

def state2tensor(state,resize):
    state = torch.from_numpy(state)
    return resize(state).unsqueeze(0).type(Tensor)

def optimize_model(model, optimizer, r_memory ):
    global last_sync
    if len(r_memory) < BATCH_SIZE:
        return
    transitions = r_memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))

    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    #reward_batch = Variable(torch.cat(Tensor(batch.reward))) #version 2
    reward_batch = Variable(torch.cat(batch.reward))# version1

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def save_checkpoint( train_portfolio , is_best, filename ='checkpoint/checkpoint.pth.tar'):
    filename = os.path.join( pwd, filename )
    bestfilename =  os.path.join( pwd,'checkpoint/model_best.pth.tar')

    torch.save( train_portfolio, filename)
    if is_best: shutil.copyfile( filename, bestfilename )

def load_checkpoint( model, optimizer,filename ='checkpoint/checkpoint.pth.tar' ,arch='UN'):
    # in reinforcement learning, the episode is not actually matter
    # we just recover the model and optimizer from blank
    # steps_done is also required to continues resume epislon Greedy policy
    global steps_done
    filename = os.path.join( pwd, filename )
    if os.path.isfile(filename):

        print( "===> Loading checkpoint from {}".format( os.path.abspath(filename) ) )
        checkpoint = torch.load(filename)

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        steps_done = checkpoint['steps_done']
        if not arch == checkpoint['arch']:
             warnings.warn('the architecture info in the checkpoint=>>> {} does not match the parameters=>>> {}'.format( checkpoint['arch'],arch ), Warning)

def maybe_folder(path):
    path = os.path.join( pwd, path )
    _ = os.mkdir(path) if not os.path.isdir( path ) else None

def create_profolio(model,optimizer,episode,arch):
        portfolio={
            'episode': episode+1,
            'arch': arch, # architecture
            'model': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'steps_done' : steps_done
            }
        return portfolio

def train(model,optimizer,env,r_memory,num_episodes,resize,plot=False,arch='UN'):

    episode_durations = []
    maybe_folder('checkpoint') # create dictory checkpoint if it is not exist

    for i_episodes in range(num_episodes):
        print( "------------episodes {}------------".format(i_episodes) )
        # initialize the state and env
        env.reset_env()
        state = env.observe_state()
        state = state2tensor(state,resize)

        # collecting the train data to the ReplayMemory and use it to train the DQN
        for t in count(): # while true
            action = select_action(state,model)

            reward, done = env.apply_action( int(action[0,0]) ) # make sure the type match the type writen in env_world_you
            reward = Tensor([reward]) # verision 1

            if not done:
                next_state = env.observe_state()
                next_state = state2tensor(next_state,resize)
            else:
                next_state = None

            #r_memory.push(state,action,next_state,reward)
            r_memory.push(state=state,action=action,next_state=next_state,reward=reward)
            state = next_state

            optimize_model(model,optimizer,r_memory)
            if done: # if return done exit the collecting process and reset the environment
                episode_durations.append( t + 1 )
                print( 'duration time: {} actions'.format(t) )
                if plot: plot_durations(model,episode_durations)
                break

        train_portfolio = create_profolio( model, optimizer, i_episodes, arch )
        save_checkpoint( train_portfolio , is_best = max(episode_durations) == t+1 )

if __name__ =="__main__":

    import models
    from gym_env import gym_env_you

    print("For testing, Using OpenAi gym as warming up environment")

    plot = False
    num_episodes = 20

    arch = 'Qestimator_resnet18'
    model = models.__dict__[arch]( num_label = 2 )
    optimizer = optim.RMSprop(model.parameters())

    # this two line is equal to model = models.Qestimator_resnet18(num_label)
    if use_cuda : model.cuda()

    r_memory= ReplayMemory( 3000, BATCH_SIZE )
    resize = T.Compose([T.ToPILImage(),
                        T.Scale(256, interpolation=Image.CUBIC),
                        T.ToTensor()])

    env = gym_env_you('DQNtest')
    if plot is True: import matplotlib.pyplt as plt

    if os.path.exists( os.path.join(pwd,'checkpoint/model_best.pth.tar') ): load_checkpoint(model, optimizer, arch = arch )
    train( model, optimizer, env, r_memory, num_episodes, resize, plot=False, arch=arch )
