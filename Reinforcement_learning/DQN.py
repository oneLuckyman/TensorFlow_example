# doi:10.1038/nature14236
# Human-level control through deep reinforcement learning 
# Algorithm 1

from collections import deque
import tensorflow as tf
import numpy as np 
import random 
import gym


env = gym.make('CartPole-v1')
EPSILON_T = 10000                   # 到达最小探索率所需要的步数
EPSILON_MIN = 0.01                  # 最小探索率
gamma = 0.9                         # 奖励衰减值
lr = 0.001                          # 学习率
n_episodes = 500                    # agent 一共玩多少局
n_steps = 1000                      # agent 每局玩多少步
TARGET_UPDATE_FREQUENCY = 10        # 每隔多少局更新一次target action-value function的权重
minibatch = 20                      # 每一次从经验池中抽取多少条经验
REWARD_BUFFER = np.empty(shape=n_episodes, dtype=list)      # 总奖励池
state_shape = (None, 4)             # 状态空间的维度
action_space_shape = 2              # 动作空间的维度


def Run_DQN(agent):
    global n_episodes               
    global n_steps                  
    global TARGET_UPDATE_FREQUENCY  
    global EPSILON_T
    global EPSILON_MIN
    global minibatch                
    global lr
    global action_space_shape
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        for t in range(n_steps):
            epsilon = np.interp(episode * n_steps + t, [0, EPSILON_T], [1, EPSILON_MIN])    # 随时间线性减少的探索率
            
            if random.random() <= epsilon:
                action = env.action_space.sample()
            else:
                action = agent.act(state)

            next_state, reward, done, info = env.step(action)
            agent.add_exp(state, action, reward, done, next_state)
            state = next_state
            episode_reward += reward        # 计算每一局游戏的总奖励

            if done:
                REWARD_BUFFER[episode] = episode_reward         # 在奖励池中添加本局分数
                print("episode: %d, epsilon: %f, score: %d" % (episode, epsilon, episode_reward))
                break 

            if len(agent.exp_replay) >= minibatch:
                batch_state, batch_action, batch_reward, batch_done, batch_next_state = agent.sample_minibatch(minibatch)
                
                # comput target action-value function
                target_q_values = agent.target_net(batch_next_state) 
                max_target_q_values = tf.reduce_max(target_q_values, axis=1)
                q_target = batch_reward + (gamma * max_target_q_values) * (1 - batch_done) 

                # comput action-value function
                # q_values = agent.action_net(batch_state)
                # a_q_values = tf.reduce_sum(q_values * tf.one_hot(batch_action, depth=action_space_shape), axis=1)

                # comput loss and gradient descent
                with tf.GradientTape() as tape:
                    loss = tf.keras.losses.mean_squared_error(
                        y_true=q_target,
                        y_pred=tf.reduce_sum(agent.action_net(batch_state) * tf.one_hot(batch_action, depth=action_space_shape), axis=1))
                grads = tape.gradient(loss, agent.action_net.variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, agent.action_net.variables))
        if (episode + 1) % TARGET_UPDATE_FREQUENCY == 0:
            agent.target_net.set_weights(agent.action_net.get_weights())        # 每隔TARGET_UPDATE_FREQUENCY 局更新一次target action-value function 的权重
            print("Avg Reward: ", np.mean(REWARD_BUFFER[:episode]))
    
    agent.target_net.save_weights('./target_net_weights')
    agent.action_net.save_weights('./action_net_weights')
    env.close()

# 一个简单的神经网络
class Model(tf.keras.Model):
    global action_space_shape

    def __init__(self):
        super().__init__()
        self.layer1 = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)
        self.layer2 = tf.keras.layers.Dense(units=action_space_shape)
    
    def call(self, inputs, training=None, mask=None):
        x = self.layer1(inputs)
        x = self.layer2(x)
        return x 

class Agent():
    global n_episodes

    def __init__(self, target_net, action_net) -> None:
        self.exp_replay = deque(maxlen=n_episodes)
        self.target_net = target_net
        self.action_net = action_net

    def act(self, state):
        q_value = self.action_net(np.expand_dims(state, axis=0))
        return tf.argmax(q_value, axis=1)[0].numpy()

    def add_exp(self, state, action, reward, done, next_state):
        self.exp_replay.append((state, action, reward, done, next_state))

    def sample_minibatch(self, minibatch_size):
        batch_state, batch_action, batch_reward, batch_done, batch_next_state = zip(*random.sample(self.exp_replay, minibatch_size))
        batch_state, batch_reward, batch_next_state, batch_done = \
                    [np.array(a, dtype=np.float32) for a in [batch_state, batch_reward, batch_next_state, batch_done]]
        batch_action = np.array(batch_action, dtype=np.int32)
        return batch_state, batch_action, batch_reward, batch_done, batch_next_state

def main():
    action_net = Model()
    action_net.build(state_shape)
    target_net = Model()
    target_net.build(state_shape)
    target_net.set_weights(action_net.get_weights())
    agent_1 = Agent(target_net, action_net)
    Run_DQN(agent=agent_1)

if __name__ == '__main__':
    main()