# -*- coding: utf-8 -*-
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 1000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        print(action_size)
        self.memory = deque(maxlen=2000)
        self.gamma = 0.995    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.0001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        LAY = 32
        LAY2 = 24
        model.add(Dense(LAY, input_dim=self.state_size, activation='relu', 
            kernel_initializer='random_normal', bias_initializer='zeros'))
        model.add(Dense(LAY, activation='relu'))
        model.add(Dense(LAY, input_dim=LAY, activation='relu',
            kernel_initializer='random_normal', bias_initializer='zeros'))
        model.add(Dense(LAY, activation='relu'))
        model.add(Dense(LAY, input_dim=LAY, activation='relu',
            kernel_initializer='random_normal', bias_initializer='zeros'))
        model.add(Dense(LAY, activation='relu'))
        model.add(Dense(LAY, input_dim=LAY, activation='relu',
            kernel_initializer='random_normal', bias_initializer='zeros'))
        model.add(Dense(LAY, activation='relu'))
        model.add(Dense(LAY, input_dim=LAY, activation='relu',
            kernel_initializer='random_normal', bias_initializer='zeros'))
        model.add(Dense(LAY, activation='relu'))
        model.add(Dense(LAY, input_dim=LAY, activation='relu',
            kernel_initializer='random_normal', bias_initializer='zeros'))
        model.add(Dense(LAY, activation='relu'))
        model.add(Dense(LAY2, input_dim=LAY, activation='relu',
            kernel_initializer='random_normal', bias_initializer='zeros'))
        model.add(Dense(LAY2, activation='relu'))
        model.add(Dense(LAY2, input_dim=LAY2, activation='relu',
            kernel_initializer='random_normal', bias_initializer='zeros'))
        model.add(Dense(LAY2, activation='relu'))
        model.add(Dense(LAY2, input_dim=LAY2, activation='relu',
            kernel_initializer='random_normal', bias_initializer='zeros'))
        model.add(Dense(LAY2, activation='relu'))
        model.add(Dense(LAY2, input_dim=LAY2, activation='relu',
            kernel_initializer='random_normal', bias_initializer='zeros'))
        model.add(Dense(LAY2, activation='relu'))
        model.add(Dense(self.action_size, activation='linear',
            kernel_initializer='random_normal', bias_initializer='zeros'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        print(len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target 
            # Filtering out states and targets for training
            states.append(state[0])
            targets_f.append(target_f[0])
        history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        # Keeping track of loss
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('Breakout-ram-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, time: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                # Logging training loss every 10 timesteps
                if time % 10 == 0:
                    print("episode: {}/{}, time: {}, loss: {:.4f}"
                        .format(e, EPISODES, time, loss))  
        state = env.reset()
        for t in range(1000):
            env.render()
            state = np.reshape(state, [1, state_size])
            state, reward, done, info = env.step(agent.act(state))
            if done:
                print(f"Episode finished after {t+1} timesteps")
                break
        if e % 10 == 0:
            agent.save(f"./save/breakout-dqn-{e}.h5")
