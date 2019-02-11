import gym.spaces
import numpy
import pandas
import math

class Game(gym.core.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self, train_data, feature_columns):
        self.train_data = train_data
        self.columns = feature_columns
        self.action_space = gym.spaces.Discrete(2)

        low_bound = numpy.array([-20]*len(self.columns))
        high_bound = numpy.array([20]*len(self.columns))
        self.observation_space = gym.spaces.Box(low=low_bound, high=high_bound)
        self.time = 0

    def step(self, action):
        time = self.time
        reward, _ = self.calc_profit(action, self.train_data["profit"][time])
        self.time += 1
        done = self.time == (len(self.train_data) - 1)
        info = {}
        observation = self.observation(time+1)
        return observation, reward, done, info

    def observation(self, time):
        return numpy.array([self.train_data[col][time] for col in self.columns])

    def get_observation(self, df, time):
        return numpy.array([df[col][time] for col in self.columns])

    def calc_profit(self, action, profit):
        """
        action
            0 stay
            1 buy
        Returns:
            reward, profit
        """
        if action == 0:
            return 0, 0
        rate = profit
        if rate >= 0:
            return rate, rate
        else:
            return rate * 3, rate

    def reset(self):
        self.time = 0
        self.profit = 0
        return self.observation(0)

    def render(self, mode):
        pass

    def close(self):
        pass

    def seed(self):
        pass