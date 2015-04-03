import numpy as np
import matplotlib.pyplot as plt
import theano

class TransitionTable:
    def __init__(self, phi_length, max_size, width, height):
        self.insert_index = -1
        self.num_entries = 0
        self.max_size = max_size
        self.phi_length = phi_length
        self.states = np.zeros((self.max_size, height, width), dtype='uint8')
        self.actions = np.zeros(self.max_size, dtype='int32')
        self.rewards = np.zeros(self.max_size, dtype=theano.config.floatX)
        self.terminals = np.zeros(self.max_size, dtype='bool')

    def add_sample(self, state, action, reward, terminal):
        self.num_entries = min(self.num_entries + 1, self.max_size)
        self.insert_index = (self.insert_index + 1) % self.max_size
        self.states[self.insert_index, ...] = state
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.terminals[self.insert_index] = terminal

    def make_phi(self, idx):
        phi = np.empty((self.phi_length, self.states.shape[1], self.states.shape[2]), dtype=theano.config.floatX)
        for i in xrange(self.phi_length):
            curr_idx = idx - i
            phi[self.phi_length-i-1,...] = self.states[curr_idx,...]
        return phi

    def phi(self, state):
        idx = self.insert_index
        phi = np.empty((self.phi_length, self.states.shape[1], self.states.shape[2]), dtype=theano.config.floatX)
        for i in xrange(self.phi_length-1):
            curr_idx = idx - i
            phi[self.phi_length-i-2,...] = self.states[curr_idx,...]
        phi[self.phi_length-1] = state
        return phi

    def get_minibatch(self, batch_size):
        states = np.empty((batch_size, self.phi_length, self.states.shape[1], self.states.shape[2]), dtype='uint8')
        next_states = np.empty((batch_size, self.phi_length, self.states.shape[1], self.states.shape[2]), dtype='uint8')
        actions = np.empty((batch_size, 1), dtype='int32')
        rewards = np.empty((batch_size, 1), dtype=theano.config.floatX)
        terminals = np.empty((batch_size, 1), dtype='bool')
        
        min_idx = self.phi_length - 1
        if self.num_entries == self.max_size:
            min_idx = 0

        i = 0
        while i < batch_size:
          idx = np.random.randint(min_idx, self.num_entries)
          if idx == self.insert_index or np.any(self.terminals[idx-(self.phi_length-1):idx+1]):
            continue
          states[i,...] = self.make_phi(idx)
          next_states[i,...] = self.make_phi((idx+1)%self.max_size)
          actions[i,...] = self.actions[idx,...]
          rewards[i,...] = self.rewards[idx,...]
          terminals[i,...] = self.terminals[idx,...]
          i += 1

        return states, actions, rewards, next_states, terminals
