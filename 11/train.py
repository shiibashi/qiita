from env import Env
from player import Player
import numpy
import pandas
from collections import deque
import random
import plot_log

class Logger(object):
    def __init__(self):
        self.train_log = []
        self.test_log = []
        self.test2_log = []
        
    def append_train_log(self, d):
        self.train_log.append(d)

    def append_test_log(self, d):
        self.test_log.append(d)

    def append_test2_log(self, d):
        self.test2_log.append(d)

    def save(self, log, filename):
        arr = numpy.array(log)
        numpy.save(filename, arr)

        
def run(player, env, logger,
        episode=100, batch_size=32, memory_size=1024, test_env=None):
    exp = deque(maxlen=memory_size)
    for e in range(episode):
        if e % 100 == 0:
            print(e, flush=True)
        done= False
        capability_list = []
        state = env.reset()
        while not done:
            action = player.get_action(state)
            next_state, reward, done, info = env.step(action)
            exp.append((state, action, reward, next_state, done))
            state = next_state
            capability_list.append(env.capability)
        if len(exp) >= memory_size:
            #return exp
            states, actions, rewards, next_states, dones = select_batch(exp, batch_size)
            player.train_model(states, actions, rewards, next_states, dones)

        logger.append_train_log(capability_list)

        if test_env is not None and e % 5 == 0:
            p = validation(player, test_env)
            logger.append_test_log(p)
            p2 = validation(player, test_env, index=1, all_length=True)
            logger.append_test2_log(p2)
            print("episode: {}, val_cap: {}, live_time: {}".format(e, p2[-1], len(p2)), flush=True)
        
def validation(player, env, index=None, all_length=False):
    done= False
    capability_list = []
    state = env.reset(index, all_length=all_length)
    while not done:
        action_prob = player.get_action(state)
        action = numpy.argmax(action_prob)
        next_state, reward, done, info = env.step(action)
        state = next_state
        capability_list.append(env.capability)
    return capability_list
    
        
def select_batch(exp, batch_size):
    tpls = random.sample(exp, batch_size)
    csv = [t[0][0] for t in tpls]
    img = [t[0][1] for t in tpls]
    actions = [t[1] for t in tpls]
    rewards = [t[2] for t in tpls]
    next_csv = [t[3][0] for t in tpls]
    next_img = [t[3][1] for t in tpls]
    dones = [t[4] for t in tpls]
    states = (_v(csv), _v(img))
    next_states = (_v(next_csv), _v(next_img))
    return states, _v(actions), _v(rewards), next_states, _v(dones)

def _v(arr_list):
    # explode
    arr = numpy.vstack([v for v in arr_list])
    return arr


def load_dataset():
    df = pandas.read_csv("dataset/feature_normalized.csv")
    train_df = df[15800:23200].reset_index(drop=True)
    test_df = df[25000:35000].reset_index(drop=True)
    return train_df, test_df


def train():
    train_df, test_df = load_dataset()

    train_env = Env(train_df)
    test_env = Env(test_df)
    action_size, csv_size, img_size = train_env.action_size, train_env.csv_size, train_env.img_size
    player = Player(action_size, csv_size, img_size)
    logger = Logger()
    run(player, train_env, logger, episode=40, batch_size=1024, memory_size=4096, test_env=test_env)

    player.actor.save_weights("model/actor_weight.h5")
    player.critic.save_weights("model/critic_weight.h5")
    plot_log.plot(logger)
    logger.save(logger.test2_log, "test_log.npy")

if __name__ == "__main__":
    train()
