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
    def __init__(self, environment, summary_dir="./"):
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
           
        self.sess = tf.Session(config=config)
        self.state = tf.placeholder(tf.float32, [None,64,64,4], 'state')
        self.advantage = tf.placeholder(tf.float32, [None, 1], 'advantage')
        self.rewards = tf.placeholder(tf.float32, [None, 1], 'd_rewards')
 
        self.dataset = tf.data.Dataset.from_tensor_slices({"state": self.state, "actions": self.actions, "rewards": self.rewards, "advantage": self.advantage})
        self.dataset = self.dataset.shuffle(buffer_size=10000)
        self.dataset = self.dataset.batch(self.minibatch)
        self.dataset = self.dataset.cache()
        self.dataset = self.dataset.repeat(4)
        self.iterator = self.dataset.make_initializable_iterator()
        batch = self.iterator.get_next()
 
        old_policy, old_policy_params = self.Actor(batch["state"], 'oldpolicy')
        policy, policy_params = self.Actor(batch["state"], 'policy')
        policy_eval, _ = self.Actor(self.state, 'policy', reuse=True)
 
        old_value, old_value_params = self.Critic(batch["state"], "oldvalue")
        self.value, value_params = self.Critic(batch["state"], "value")
        self.value_eval, _ = self.Critic(self.state, 'value', reuse=True)
 
        self.sample_action = tf.squeeze(policy_eval.sample(1), axis=0, name="sample_action")
        self.global_step = tf.train.get_or_create_global_step()
        self.saver = tf.train.Saver()
 
        with tf.variable_scope('loss'):
            with tf.variable_scope('actor'):
                ratio = tf.maximum(policy.prob(batch["actions"]), 1e-6) / tf.maximum(old_policy.prob(batch["actions"]), 1e-6)
                ratio = tf.clip_by_value(ratio, 0, 10)
                surr1 = batch["advantage"] * ratio
                surr2 = batch["advantage"] * tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon)
                loss_policy = -tf.reduce_mean(tf.minimum(surr1, surr2))
                tf.summary.scalar("loss", loss_policy)
 
            with tf.variable_scope('critic'):
                loss_actor = tf.reduce_mean(tf.square(self.value - batch["rewards"])) * 0.5
                tf.summary.scalar("loss", loss_actor)
 
            with tf.variable_scope('entropy'):
                entropy = policy.entropy()
                pol_entpen = -self.beta * tf.reduce_mean(entropy)
 
            loss = loss_policy + loss_actor * self.critic_coefficient + pol_entpen
            tf.summary.scalar("total", loss)
 
        with tf.variable_scope('train'):
            opt = tf.train.AdamOptimizer(self.learning_rate)
            self.trainer = opt.minimize(loss, global_step=self.global_step, var_list=policy_params + value_params)

        with tf.variable_scope('update_old'):
            self.update_old_policy_op = [oldp.assign(p) for p, oldp in zip(policy_params, old_policy_params)]
            self.update_old_value_op = [oldp.assign(p) for p, oldp in zip(value_params, old_value_params)]
 
        self.writer = tf.summary.FileWriter(summary_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
 
        tf.summary.scalar("value", tf.reduce_mean(self.value))
        tf.summary.scalar("policy_entropy", tf.reduce_mean(entropy))
        tf.summary.scalar("sigma", tf.reduce_mean(policy.stddev()))
        self.board = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))
 
    def save_model(self, model_path, step=None):
        save_path = self.saver.save(self.sess, os.path.join(model_path, "model.ckpt"), global_step=step)
        return save_path
 
    def train(self, state, action, reward, advantage):
        self.sess.run([self.update_old_policy_op, self.update_old_value_op, self.iterator.initializer],
                      feed_dict={self.state: state, self.actions: action, self.rewards: reward, self.advantage: advantage})
        while True:
            try:
                summary, step, _ = self.sess.run([self.board, self.global_step,self.trainer])
            except tf.errors.OutOfRangeError:
                break
            
        print("Sum of Step %i" % step)
        return summary
 
    def Actor(self, state_in, name, reuse=False):
        w_reg = tensorflow.keras.regularizers.l2(self.l2_regular)
 
        with tf.variable_scope(name, reuse=reuse):
            input_ = tf.cast(state_in, tf.float32)
            c1 = tf.layers.conv2d(inputs=input_, filters=16, kernel_size=8, strides=4, activation=tf.nn.leaky_relu, padding="valid")
            c2 = tf.layers.conv2d(inputs=c1, filters=32, kernel_size=3, strides=2, activation=tf.nn.leaky_relu, padding="valid")
            state_in = tf.layers.flatten(c2)
  
            mu = tf.layers.dense(state_in, self.action_size, tf.nn.tanh, kernel_regularizer=w_reg, name="policy_mu")
            log_sigma = tf.get_variable(name="policy_sigma", shape=self.action_size, initializer=tf.zeros_initializer())
            dist = tf.distributions.Normal(loc=mu, scale=tf.exp(log_sigma))
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return dist, params
 
    def Critic(self, state_in, name, reuse=False):
        w_reg = tensorflow.keras.regularizers.l2(self.l2_regular)
 
        with tf.variable_scope(name, reuse=reuse):
            input_ = tf.cast(state_in, tf.float32)
            c1 = tf.layers.conv2d(inputs=input_, filters=16, kernel_size=8, strides=4, activation=tf.nn.leaky_relu)
            c2 = tf.layers.conv2d(inputs=c1, filters=32, kernel_size=3, strides=2, activation=tf.nn.leaky_relu)
            state_in = tf.layers.flatten(c2)
 
            value = tf.layers.dense(state_in, 1, kernel_regularizer=w_reg, name="value_output")
 
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return value, params
 
    def eval(self, state):
        action, value = self.sess.run([self.sample_action, self.value_eval], {self.state: state[np.newaxis, :]})
        return action[0], np.squeeze(value)
 
 
def discount(delta, gamma, done_array=None):
    if done_array is None:
        return scipy.signal.lfilter([1], [1, -gamma], delta[::-1], axis=0)[::-1]
    else:
        ad, adv = 0, []
        dones_reversed = done_array[1:][::-1]
        for step, dt in enumerate(reversed(delta)):
            ad = dt + gamma * ad * (1 - dones_reversed[step])
            adv.append(ad)
        return np.array(adv)[::-1]
 
 
if __name__ == '__main__':
    Env = 'CarRacing-v0'
    max_episodes = 10000
    gamma = 0.99
    lmbda = 0.95
    epoch_batch = 2048
    Timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    Sum_dir = os.path.join("./PPO", Timestamp)
 
    env = gym.make(Env)
    env = wrappers.Monitor(env, os.path.join(Sum_dir, Env), video_callable=None)
    ppo = PPO(env, Sum_dir)
    stacked_frames = deque(maxlen=4)

    timestep = 0
    buffer_state, buffer_action, buffer_reward, buffer_value, buffer_done = [], [], [], [], []

    for episode in range(max_episodes + 1):
        state = env.reset()
        state_ = preprocess_frame(state)
        for i in range(4):
            stacked_frames.append(state_)

        ep_reward, ep_timestep, done = 0, 0, False
        check_reward = deque(maxlen=100)
        while True:
            stacked_states = np.concatenate(stacked_frames, axis=2)
            #stacked_states = np.expand_dims(stacked_states, axis=0)
            action, value = ppo.eval(stacked_states)
            
            #train ppo
            if timestep == epoch_batch:
                value_final = [value * (1 - done)]
                rewards = np.array(buffer_reward)
                values = np.array(buffer_value + value_final)
                dones = np.array(buffer_done + [done])

                #GAE
                delta = rewards + gamma * values[1:] * (1 - dones[1:]) - values[:-1]
                advantage = discount(delta, gamma * lmbda, dones)
                returns = advantage + np.array(buffer_value)
                advantage = (advantage - advantage.mean()) / np.maximum(advantage.std(), 1e-6)
 
                batch_state, batch_action, batch_reward, batch_advantage = np.reshape(buffer_state, (timestep,) + (64,64,4)), np.vstack(buffer_action), \
                                   np.vstack(returns), np.vstack(advantage)
 
                Summary_PPO = ppo.train(batch_state, batch_action, batch_reward, batch_advantage)
                buffer_state, buffer_action, buffer_reward, buffer_value, buffer_done = [], [], [], [], []
                timestep = 0
 
            if done:
                print('Episode: %d' % episode, "Reward: %.2f" % ep_reward)
 
                Summary_reward = tf.Summary()
                Summary_reward.value.add(tag="Reward", simple_value=ep_reward)
                try:
                    ppo.writer.add_summary(Summary_PPO, episode)
                except NameError:
                    pass
                ppo.writer.add_summary(Summary_reward, episode)
                ppo.writer.flush()
 
                if episode % 200 == 0 and episode != 0:
                    path = ppo.save_model(Sum_dir, episode)
                break
 
            buffer_state.append(stacked_states)
            buffer_action.append(action)
            buffer_value.append(value)
            buffer_done.append(done)

            action = np.clip(action, ppo.action_bound_low, ppo.action_bound_high)
            state, reward, done, _ = env.step(action)
            state_ = preprocess_frame(state)
            stacked_frames.append(state_)

            if np.mean(state[:,:,1]) > 185.0:
                reward = reward - 0.05
            
            check_reward.append(reward)
            buffer_reward.append(reward)
 
            ep_reward += reward
            ep_timestep += 1
            timestep += 1

    env.close()
