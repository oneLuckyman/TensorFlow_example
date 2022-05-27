# https://blog.csdn.net/qq_40041064/article/details/105047546

import tensorflow as tf 
import numpy as np 
import gym 
import random
from collections import deque

num_episodes = 500                  # 游戏训练的总episode数
num_exploration_episodes = 100      # 探索过程所占的episode数量
max_len_episode = 1000              # 每个episode的最大回合数
batch_size = 32                     # 批次大小
learning_rate = 0.001               # 学习率
gamma = 1.0                         # 折扣因子
initial_epsilon = 1.0               # 探索起始时的探索率
final_epsilon = 0.01                # 探索终止时的探索率

class QNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=24, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=24, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=2)
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x
    
    def predict(self, inputs):
        q_values = self(inputs)
        return tf.argmax(q_values, axis=1)