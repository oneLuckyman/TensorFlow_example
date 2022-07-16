# doi:10.1038/nature14236
# Human-level control through deep reinforcement learning 
# Algorithm 1

from pickletools import optimize
import tensorflow as tf
import numpy as np 
import random 
import gym


env = gym.make('CartPole-v1')
EPSILON_T = 10000
EPSILON_MIN = 0.01
gamma = 0.9
lr = 0.001
n_episodes = 500
n_steps = 1000
TARGET_UPDATE_FREQUENCY = 10
minibatch = 20
REWARD_BUFFER = np.empty(shape=n_episodes, dtype=list)
action_net, target_net = 0, 0


def Run_DQN(agent):
    global n_episodes               # agent 一共玩多少局
    global n_steps                  # agent 每局玩多少步
    global TARGET_UPDATE_FREQUENCY  # 每隔多少局更新一次target action-value function的权重
    global minibatch                # 每一次从经验池中抽取多少条经验
    global lr
    optimizer = tf.kears.optimizers.Adam(learning_rate = lr)
    for episode in n_episodes:
        state = env.reset()
        for t in n_steps:
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
                print("episode: %d, epsilon: %d, score: %d" % (episode, epsilon, episode_reward))
                break 

            if len(agent.exp_replay) >= minibatch:
                batch_state, batch_action, batch_reward, batch_done, batch_next_state = agent.sample_minibatch(minibatch)
                
                # comput target action-value function
                target_q_values = agent.target_net(batch_next_state) 
                max_target_q_values = tf.reduce_max(target_q_values, axis=1)
                y = batch_reward + (gamma * max_target_q_values) * (1 - batch_done) 

                # comput action-value function
                q_values = agent.action_net(batch_state)
                a_q_values = tf.reduce_sum(q_values * tf.one_hot(batch_action, depth=2), axis=1)

                # comput loss and gradient descent
                with tf.GradientTape() as tape:
                    loss = tf.keras.losses.mean_squared_error(
                        y_true=y,
                        y_pred=a_q_values)
                grads = tape.gradient(loss, agent.action_net.variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, agent.action_net.variables))
        if episode % TARGET_UPDATE_FREQUENCY == 0:
            agent.target_net.set_weights(agent.action_net.get_weights())        # 每隔TARGET_UPDATE_FREQUENCY 局更新一次target action-value function 的权重
            print("Avg Reward: ", np.mean(REWARD_BUFFER[:episode]))
    env.close()
