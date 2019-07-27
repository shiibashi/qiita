import gym
import numpy
import pandas
import random
import os

class Systra(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, mode="train", random_data=True):
        self.data_dir = "data/sample_train" if mode=="train" else "data/sample_test"
        self.random_data = random_data
        self.filename_list = os.listdir(self.data_dir)
        
        self.columns = ["f1", "f2"]
        
        lb = numpy.array([1, 1, -1])
        ub = numpy.array([0, 0, 2])
        self.observation_space = gym.spaces.Box(lb, ub, dtype=numpy.float32)
        self.action_space = gym.spaces.Discrete(3) # 0: stay, 1: long, 2: short
        
        #self.seed()
        self.state = None
        
        self.time = None
        self.code = None
        self.code_data = None
        self.power = None
                
    def seed(self, seed=None):
        self.numpy_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
        
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        reward = self.reward(action, self.time, self.code_data)
        self.time += 1
        self.cut_line += 0.001
        done = self.done_flag(self.time, self.code_data)
        next_obj = self.observation(self.time, self.code_data)
        return next_obj, reward, done, {}
        
    def reset(self):
        if self.random_data:
            filename = random.sample(self.filename_list, 1)[0]
        else:
            filename = "0500.csv"
        code_data = self._load_code_data(filename)
        #code_data = self.all_df.sample(300).reset_index(drop=True)
        self.time = 0
        self.code = filename.split(".")[0]
        self.code_data = code_data
        self.power = 1
        self.cut_line = 1
        return self.observation(self.time, self.code_data)

    def observation(self, time, df):
        feature = numpy.array([df[col][time] for col in self.columns])
        status = numpy.array([self.power - self.cut_line])
        s = len(feature) + len(status)
        obs = numpy.concatenate([feature, status]).reshape(1, s)
        #return numpy.array([[df[col][time] for col in self.columns]])
        return obs
        
    def reward(self, action, time, df):
        profit = self.code_data["profit"][time]
        if action == 2: # short
            action = -1
        
        if self.power - self.cut_line >= 0.5 and profit * action < 0:
            self.power += profit * action * 10
        else:
            self.power += profit * action
        return max(0, self.power - self.cut_line)

        
    def done_flag(self, time, df):
        if time < len(df) - 1 and self.power >= self.cut_line - 0.02:
            return False
        else:
            return True
        
    def render(self, mode="human"):
        pass
        
    def close(self):
        pass
        
    def _load_code_data(self, filename):
        df = pandas.read_csv("{}/{}".format(self.data_dir, filename))
        return df