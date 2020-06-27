import tensorflow
import tensorflow.compat.v1 as tf
import numpy as np
import gym
import os
import cv2
import scipy.signal
from collections import deque
from gym import wrappers
from datetime import datetime
from time import time
tf.compat.v1.disable_eager_execution()

def preprocess_frame(frame):
    frame = frame[0:84, :, :]
    frame = cv2.resize(frame, (64,64))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = np.asarray(frame, np.float32) / 255
    frame = np.reshape(frame, (64,64,1))
    return frame

class PPO(object):
    def __init__(self, environment):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': True})
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
 
        self.state_size, self.action_size = environment.observation_space.shape, environment.action_space.shape[0]
        self.action_bound_high = environment.action_space.high
        self.action_bound_low = environment.action_space.low
        self.actions = tf.placeholder(tf.float32, [None, self.action_size], 'action')
        self.beta = 0.01
        self.learning_rate = 0.0001
        self.minibatch = 32
        self.epsilon = 0.21
        self.critic_coefficient = 0.5
        self.l2_regular = 0.001
        
        self.state = tf.placeholder(tf.float32, [None,64,64,4], 'state')
        self.advantage = tf.placeholder(tf.float32, [None, 1], 'advantage')
        self.rewards = tf.placeholder(tf.float32, [None, 1], 'd_rewards')
 
        
        policy_eval, _ = self.Actor(self.state, 'policy')
 
        
        self.value_eval, _ = self.Critic(self.state, 'value')
 
        self.sample_op = tf.squeeze(policy_eval.sample(1), axis=0, name="sample_action")          
  
        self.saver = tf.train.Saver()
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        tf.reset_default_graph()
        self.saver.restore(self.sess, "./model.ckpt-10000")
        
  
    
    def Actor(self, state_in, name, reuse=False):
        w_reg = tensorflow.keras.regularizers.l2(self.l2_regular)
 
        with tf.variable_scope(name, reuse=reuse):
            scaled = tf.cast(state_in, tf.float32)
            conv1 = tf.layers.conv2d(inputs=scaled, filters=16, kernel_size=8, strides=4, activation=tf.nn.leaky_relu, padding="valid")
            conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=3, strides=2, activation=tf.nn.leaky_relu, padding="valid")
            state_in = tf.layers.flatten(conv2)
  
            mu = tf.layers.dense(state_in, self.action_size, tf.nn.tanh, kernel_regularizer=w_reg, name="policy_mu")
            log_sigma = tf.get_variable(name="policy_sigma", shape=self.action_size, initializer=tf.zeros_initializer())
            dist = tf.distributions.Normal(loc=mu, scale=tf.exp(log_sigma))
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return dist, params
 
    def Critic(self, state_in, name, reuse=False):
        w_reg = tensorflow.keras.regularizers.l2(self.l2_regular)
 
        with tf.variable_scope(name, reuse=reuse):
            scaled = tf.cast(state_in, tf.float32)
            conv1 = tf.layers.conv2d(inputs=scaled, filters=16, kernel_size=8, strides=4, activation=tf.nn.leaky_relu)
            conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=3, strides=2, activation=tf.nn.leaky_relu)
            state_in = tf.layers.flatten(conv2)
 
            value = tf.layers.dense(state_in, 1, kernel_regularizer=w_reg, name="value_output")
 
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return value, params
 
    def eval(self, state):
        action, value = self.sess.run([self.sample_op, self.value_eval], {self.state: state[np.newaxis, :]})
        return action[0], np.squeeze(value)
 
 
def discount(x, gamma, done_array=None):
    if done_array is None:
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
    else:
        y, adv = 0, []
        dones_reversed = done_array[1:][::-1]
        for step, dt in enumerate(reversed(x)):
            y = dt + gamma * y * (1 - dones_reversed[step])
            adv.append(y)
        return np.array(adv)[::-1]
 
 
if __name__ == '__main__':
    Env = 'CarRacing-v0'
    max_episodes = 10000
    gamma = 0.99
    lmbda = 0.95
    epoch_batch = 2048 
 
    env = gym.make(Env)
    
    ppo = PPO(env)
    stacked_frames = deque(maxlen=4)

     
    state = env.reset()
    state_ = preprocess_frame(state)
    for i in range(4):
        stacked_frames.append(state_)

    done = False
    while True:
        env.render("human")
        stacked_states = np.concatenate(stacked_frames, axis=2)
            #stacked_states = np.expand_dims(stacked_states, axis=0)
        action, value = ppo.eval(stacked_states)
        if done:
            break
        action = np.clip(action, ppo.action_bound_low, ppo.action_bound_high)
        state, reward, done, _ = env.step(action)
        state_ = preprocess_frame(state)
        stacked_frames.append(state_)

