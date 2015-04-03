#!/usr/bin/env python
# vim: tabstop=2 expandtab shiftwidth=2 softtabstop=2
from __future__ import division
import cPickle
import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import sys
import theano
import time
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader
from rlglue.types import Action, Observation
from rlglue.utils import TaskSpecVRLGLUE3
from network import DeepQLearner
from transition import TransitionTable
sys.setrecursionlimit(10000)

IMAGE_WIDTH = 160
IMAGE_HEIGHT = 210
RESIZED_WIDTH = 84
RESIZED_HEIGHT = 84

class DeepQLearningAgent(Agent):
  def __init__(self, prefix, network_file):
    self.prefix = prefix
    self.network_file = network_file

  def agent_init(self, spec):
    taskSpec = TaskSpecVRLGLUE3.TaskSpecParser(spec)
    if taskSpec.valid:
      self.num_actions = taskSpec.getIntActions()[0][1] + 1
    else:
      raise "Invalid task spec"
    self.last_observation = Observation()

    self.batch_size = 32 # batch size for SGD
    self.ep_start = 1 # initial value of epsilon in epsilon-greedy exploration
    self.ep = self.ep_start # exploration probability
    self.ep_end = 0.1 # final value of epsilon in epsilon-greedy exploration
    self.ep_endt = 1000000 # number of frames over which epsilon is linearly annealed
    self.episode_qvals = []
    self.all_qvals = []
    self.learn_start = 0 # number of steps after which learning starts
    self.is_testing = False
    self.replay_memory = 1000000
    self.phi_length = 4 # number of most recent frames for input to Q-function
    self.reset_after = 10000 # replace Q_hat with Q after this many steps
    self.step_counter = 0
    self.episode_counter = 0
    self.total_reward = 0
    self.qvals = []

    self.train_table = TransitionTable(self.phi_length, self.replay_memory, RESIZED_WIDTH, RESIZED_HEIGHT)
    self.test_table = TransitionTable(self.phi_length, self.phi_length, RESIZED_WIDTH, RESIZED_HEIGHT)
    if self.network_file is None:
      self.network = DeepQLearner(RESIZED_WIDTH, RESIZED_HEIGHT, self.num_actions, self.phi_length, self.batch_size)
    else:
      self.network = cPickle.load(open(self.network_file))

  def agent_start(self, observation):
    this_int_action = np.random.randint(0, self.num_actions)
    return_action = Action()
    return_action.intArray = [this_int_action]
    self.start_time = time.time()
    self.batch_counter = 0
    self.last_action = 0
    self.losses = []

    self.last_img = self.resize_image(observation.intArray)

    return return_action

  def agent_step(self, reward, observation):
    self.step_counter += 1
    self.total_reward += reward
    cur_img = self.resize_image(observation.intArray)

    if self.is_testing:
      int_action = self.choose_action(self.test_table, cur_img, np.clip(reward, -1, 1), testing_ep=0.05)
    else:
      if self.step_counter % self.reset_after == 0:
        self.network.reset_q_hat()

      int_action = self.choose_action(self.train_table, cur_img, np.clip(reward, -1, 1), testing_ep=None)
      if self.train_table.num_entries > max(self.learn_start, self.batch_size):
        states, actions, rewards, next_states, terminals = self.train_table.get_minibatch(self.batch_size)
        loss, qvals = self.network.train(states, actions, rewards, next_states, terminals)
        self.losses.append(loss)
        self.qvals.append(np.mean(qvals))
        self.batch_counter += 1

    return_action = Action()
    return_action.intArray = [int_action]

    self.last_action = int_action
    self.last_img = cur_img

    return return_action

  def choose_action(self, table, cur_img, reward, testing_ep):
    table.add_sample(self.last_img, self.last_action, np.clip(reward, -1, 1), terminal=False)

    self.ep = testing_ep or (self.ep_end +
                max(0, (self.ep_start - self.ep_end) * (self.ep_endt -
                max(0, self.step_counter - self.learn_start))/self.ep_endt))

    return self.epsilon_greedy(table, cur_img, testing_ep)

  def epsilon_greedy(self, table, cur_img, testing_ep):
    self.ep = testing_ep or (self.ep_end +
                max(0, (self.ep_start - self.ep_end) * (self.ep_endt -
                max(0, self.step_counter - self.learn_start))/self.ep_endt))
    if np.random.random() < self.ep:
      return np.random.randint(0, self.num_actions)
    else:
      return self.greedy(table, cur_img)

  def greedy(self, table, cur_img):
    if self.step_counter >= self.phi_length:
      phi = table.phi(cur_img)

      int_action = self.network.choose_action(phi)
    else:
      int_action = np.random.randint(0, self.num_actions)
    return int_action

  def agent_end(self, reward):
    self.step_counter += 1
    self.episode_counter += 1
    self.total_reward += reward
    if not self.is_testing:
      self.train_table.add_sample(self.last_img, self.last_action, np.clip(reward, -1, 1), terminal=True)

    time_taken = time.time() - self.start_time
    if len(self.losses) > 0:
      print "Episode: %d" % self.episode_counter, "Rate: %.2f" % (self.batch_counter / time_taken), "Ep: %.4f" % self.ep, " Avg Loss: %.4f" % np.mean(self.losses), " Avg Qval: %.4f" % np.mean(self.qvals), " Avg Reward: ", (self.total_reward / self.episode_counter)

  def agent_cleanup(self):
    pass

  def agent_message(self, inMessage):
    if inMessage.startswith("episode_end"):
      self.agent_end(0)
    elif inMessage.startswith("finish_epoch"):
      self.episode_qvals.append(np.mean(self.qvals))
      self.all_qvals += self.qvals
      self.qvals = []
      self.total_reward = 0
      self.episode_counter = 0

      epoch = int(inMessage.split(' ')[1])
      cPickle.dump(self.network, open('blobs/%s_%d.pkl' % (self.prefix, epoch), 'w'), -1)
      print "ep: ", self.ep, " step_counter: ", self.step_counter
      self.render_plot()
    elif inMessage.startswith("start_testing"):
      self.is_testing = True
      self.total_reward = 0
      self.episode_counter = 0
    elif inMessage.startswith("finish_testing"):
      self.is_testing = False
    else:
      print "Got: ", inMessage

  def render_plot(self):
    try:
      plt.figure()
      plt.plot(xrange(len(self.all_qvals)), self.all_qvals)
      plt.savefig('plots/%s_all_qvals.png' % self.prefix)
      plt.close()

      plt.figure()
      plt.plot(xrange(len(self.episode_qvals)), self.episode_qvals)
      plt.savefig('plots/%s_episode_qvals.png' % self.prefix)
      plt.close()
    except:
      print "Failed to render plots"

  def resize_image(self, observation):
    image = observation[128:].reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    image = np.array(image, dtype='uint8')

    offset = 10 # remove ACTIVISION logo
    width = RESIZED_WIDTH
    height = int(round(float(IMAGE_HEIGHT) * RESIZED_HEIGHT / IMAGE_WIDTH))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    image = image[height-offset-RESIZED_HEIGHT:height-offset, :]

    if False:
      plt.figure()
      plt.imshow(image, cmap=cm.Greys_r)
      plt.show()

    return image

if __name__== '__main__':
  prefix = sys.argv[1] if len(sys.argv) > 1 else ""
  network_file = int(sys.argv[2]) if len(sys.argv) > 2 else None
  AgentLoader.loadAgent(DeepQLearningAgent(prefix, network_file))
