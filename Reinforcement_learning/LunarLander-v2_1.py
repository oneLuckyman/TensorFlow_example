import matplotlib.pyplot as plt
from requests import session
import tensorflow as tf
import numpy as np
import threading 
import gym 

episodes = 2000
gamma = 0.9 
learning_rate = 0.001
num_workers = 3 

game = 'LunarLander-v2'
state_shape = (None, 8)
num_actions = 4

EPISODE = 0
class Model(tf.keras.models.Model):
    def __init__(self, num_actions):
        super(Model, self).__init__()
        self.layer1 = tf.keras.layers.Dense(128, activation='relu')
        self.layer2 = tf.keras.layers.Dense(128, activation='relu')
        self.actor_layer = tf.keras.layers.Dense(num_actions, activation='linear')
        self.critic_layer = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs, training=None, mask=None):
        x = self.layer1(inputs)
        x = self.layer2(x)
        actor_output = self.actor_layer(x)
        critic_output = self.critic_layer(x)
        return actor_output, critic_output

def run_worker(global_model, global_optimizer):
    global game 
    global EPISODE
    global rewards_plt
    global losses_plt
    global state_shape
    global gamma
    global num_actions

    env = gym.make(game)
    model = Model(num_actions)
    model.build(state_shape)
    
    while EPISODE < episodes:
        all_rewards = 0
        states, values, act_probs, actions, rewards = [], [], [], [], []

        state = np.array([env.reset()], dtype=np.float)

        model.set_weights(global_model.get_weights())

        for steps in range(5000):
            states.append(state[0])
            act_prob, value = model(state)
            act_probs.append(act_prob[0])
            values.append(value[0])
            policy = tf.nn.softmax(act_prob)
            action = np.random.choice(num_actions, p=policy.numpy()[0])
            actions.append(action)
            state, reward, done, info = env.step(action)
            rewards.append(reward)
            all_rewards += reward
            state = np.array([state], dtype=np.float)
            if done:
                break

        rewards_plt.append(all_rewards)
        
        with tf.GradientTape() as tape:
            _values = []
            for i in range(len(values) - 1):
                _values.append([rewards[i] * gamma + values[i+1][0]])
            _values.append([rewards[-1]])
            advantages = np.array(_values) - np.array(values)
            advantages = np.reshape(advantages, newshape=(-1))

            actions_onehot = np.eye(num_actions)[actions]
            act_prob, value = model(np.array(states, dtype=np.float))
            policy = tf.nn.softmax(act_prob)

            losses = advantages * tf.nn.softmax_cross_entropy_with_logits(labels=actions_onehot, logits=act_prob) + \
                        0.5 * tf.reshape((value - _values) ** 2, shape=(-1)) + \
                        0.01 * tf.reduce_mean(policy * tf.math.log(policy + 1e-20), axis=-1)

            grad = tape.gradient(tf.reduce_mean(losses), model.trainable_variables)
            global_optimizer.apply_gradients(zip(grad, global_model.trainable_variables))

            print('episode"{}"; rewards"{}"; losses"{}"'.format(EPISODE + 1, all_rewards, tf.reduce_mean(losses)))
        losses_plt.append(tf.reduce_mean(losses))
        EPISODE += 1

if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True 
    session = tf.compat.v1.InteractiveSession(config=config)

    global_model = Model(num_actions)
    global_model.build(input_shape=state_shape)
    global_optimizer = tf.keras.optimizers.Adam(learning_rate)

    rewards_plt = []
    losses_plt = []

    threads = []
    for _ in range(num_workers):
        p = threading.Thread(target=run_worker, args=[global_model, global_optimizer])
        p.start()
        threads.append(p)
    for p in threads: p.join()

    global_model.save_weights('./A3C_LunarLander_2e3_epochs.h5')
    plt.plot(rewards_plt)
    plt.show()
    plt.plot(losses_plt)
    plt.show()