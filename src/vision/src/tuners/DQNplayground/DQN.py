#!/usr/bin/env python

from collections import namedtuple

from itertools import count
from copy import deepcopy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

import pickle

from model import Qestimator
from gym_env import gym_env_you

use_cuda = torch.cuda.is_available()
if use_cuda is False: raise Exception('Check your cuda!')

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200


Transition = namedtuple(typename="Transition",field_names=("state","action","next_state","reward") )
# collections.namedtuple(typename, field_names[, verbose=False][, rename=False])
class ReplayMemory(object):

    def __init__(self,capacity,batch_size=1):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.batch_size = batch_size

    def push(use_tuple=False,ntuple=None,*args1,**args2):
        if len(self.memory)< self.capacity:
            self.memory.append(None)
        if not use_tuple:
            self.memory[self.position] = Transition(*args1,**args2)
        else:
            self.memory[self.position] = ntuple

    def sample(self,batch_size=None):
        batch_size = batch_size if batch_size!=None else self.batch_size
        return random.sample(self.memory, batch_size)

    def dump(path):
        with open(path,'wb'):
            pickle.dump(self.memory,path,-1) # use highest protocal

    def load(path):
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

def select_action(state):
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

global last_sync = 0
def optimize_model(model,r_memory):
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
    reward_batch = Variable(torch.cat(Tensor(batch.reward))) version 2
    #reward_batch = Variable(torch.cat(batch.reward)) version1

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





def train(model,env,r_memory,episode_durations,plot=False):
    num_episodes = 10
    episode_durations = []

    for i_episodes in range(num_episodes):
        # initialize the state and env
        reset_env(env)
        state = observe_state(env)

        for t in count(): # while true
            action = select_action(state)
            reward, done = apply_action(env,action)
            # reward = Tensor([reward]) # verision 1
            if not done:
                next_state = observe_state()
            else:
                next_state = None

            r_memory.push("state"=state,"action"=action,"next_state"=next_state,"reward"=reward)
            state = next_state()

            optimize_model()
            if done:
                episode_durations.append( t+1 )
                if plot: plot_durations(model,episode_durations)
                break

def run():


if __name__ =="__main__":
    model = Qestimator()
    r_memory= ReplayMemory( 5000, BATCH_SIZE )
    resize = T.Compose([T.ToPILImage(),
                        T.Scale(40, interpolation=Image.CUBIC),
                        T.ToTensor()])
