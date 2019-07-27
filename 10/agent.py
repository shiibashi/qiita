import numpy


import keras
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Concatenate, Dropout
from keras.layers.normalization import BatchNormalization
from keras import Input, Sequential
from keras.optimizers import SGD



class Agent(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        
        actor, critic = self.build()
        self.actor = actor
        self.critic = critic
    
    
    def build(self):
        actor = Sequential()
        actor.add(Dense(2, input_dim=self.state_size, activation='relu'))
        actor.add(Dense(self.action_size, activation='softmax'))
        actor.summary()
        actor.compile(SGD(), loss='categorical_crossentropy')

        critic = Sequential()
        critic.add(Dense(2, input_dim=self.state_size, activation='relu'))
        critic.add(Dense(self.value_size, activation='linear'))
        critic.summary()
        critic.compile(SGD(), loss="mse")
        return actor, critic
    

    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        noise = numpy.random.uniform(0, 1, self.action_size) * 0.5
        p = (policy + noise) / (policy + noise).sum()
        return numpy.random.choice(self.action_size, 1, p=p)[0]

    def get_action_prob(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return policy

        
    def train_model(self, states, actions, rewards, next_states, dones):
        length = len(states)
        target = numpy.zeros((length, self.value_size))
        advantages = numpy.zeros((length, self.action_size))
        #print(states)
        values = self.critic.predict(states)
        next_values = self.critic.predict(next_states)

        for i, done in enumerate(dones):
            action = actions[i]
            reward = rewards[i]
            value = values[i]
            next_value = next_values[i]
            if done:
                advantages[i][action] = reward - value
                target[i][0] = reward
            else:
                r = 0.9
                advantages[i][action] = reward + r * (next_value) - value
                target[i][0] = reward + r * next_value

        self.actor.fit(states, advantages, epochs=1, verbose=0)
        self.critic.fit(states, target, epochs=1, verbose=0)
