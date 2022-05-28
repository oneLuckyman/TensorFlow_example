# https://blog.csdn.net/qq_40041064/article/details/105047546

from pickletools import optimize
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

if __name__ == '__main__':
    env = gym.make('CartPole-v1')                           # 实例化一个游戏环境，参数为游戏名称
    model = QNetwork()                                      # 实例化一个Q网络
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    replay_buffer = deque(maxlen=10000)                     # 使用一个 deque 作为经验回放池
    epsilon = initial_epsilon
    for episode_id in range(num_episodes):
        state = env.reset()                                 # 初始化环境，获得初始状态
        epsilon = max(                                      # 计算当前探索率
            initial_epsilon * (num_exploration_episodes - episode_id) / num_exploration_episodes,
            final_epsilon)
        for t in range(max_len_episode):
            env.render()                                    # 对当前帧进行渲染，绘图到屏幕
            if random.random() < epsilon:                   # epsilon-greedy 探索策略，以 epsilon 的概率随机选择动作
                action = env.action_space.sample()          # 选择随机动作（探索）
            else:
                action = model.predict(np.expand_dims(state, axis=0)).numpy()   # 选择模型计算出的 Q 值最大的动作
                action = action[0]

        