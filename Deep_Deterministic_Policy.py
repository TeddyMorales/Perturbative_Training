import numpy as np
from collections import deque
import random


class ReplayBuffer(object):

    def __init__(self, buffer_size):
        pass

    def add(self, s, a, r, t, s2):
        pass

    def size(self):
        pass

    def sample_batch(self, batch_size):
        '''
        batch size specifies the number of experiences to add to the batch. If the replay buffer has less than batch_size elements, simply return all of the elements within the buffer. Generally, you'll want to wait until the buffer has at least batch_size elements before beginning to sample from it.
        '''
        pass

    def clear(self):
        pass

class ActorNetwork(object):

    def create_actor_network(self):
        pass            

class CriticNetwork(object):

    def create_critic_network(self):
        pass
