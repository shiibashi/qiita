#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import multiprocessing
import tensorflow
import argparse
import time

from actor import Actor
from learner import Learner
import dataset
from game import Game

def actor_work(args, queues, env, num):
    # with tensorflow.device('/cpu:0'):
    sess = tensorflow.InteractiveSession()
    actor = Actor(args, queues, num, sess, env, param_copy_interval=2000, send_size=200)
    actor.run()

def leaner_work(args, queues, env):
    # with tensorflow.device('/gpu:0'):
    sess = tensorflow.InteractiveSession()
    leaner = Learner(args, queues, sess, env, batch_size=126)
    leaner.run()

def _parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_actors', type=int, default=2, help='number of Actors')
    parser.add_argument('--env_name', type=str, default='Alien-v0', help='Environment of Atari2600 games')
    parser.add_argument('--train', type=int, default=1, help='train mode or test mode')
    parser.add_argument('--test_gui', type=int, default=0, help='decide whether you use GUI or not in test mode')
    parser.add_argument('--load', type=int, default=0, help='loading saved network')
    parser.add_argument('--network_path', type=str, default=0, help='used in loading and saving (default: \'saved_networks/<env_name>\')')
    parser.add_argument('--replay_memory_size', type=int, default=2000000, help='replay memory size')
    parser.add_argument('--initial_memory_size', type=int, default=20000, help='Learner waits until replay memory stores this number of transition')
    parser.add_argument('--num_episodes', type=int, default=10000, help='number of episodes each agent plays')
    parser.add_argument('--frame_width', type=int, default=84, help='width of input frames')
    parser.add_argument('--frame_height', type=int, default=84, help='height of input frames')
    parser.add_argument('--state_length', type=int, default=4, help='number of input frames')
    parser.add_argument('--n_step', type=int, default=3, help='n step bootstrap target')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')

    args = parser.parse_args()
    return args
    
def _train(args, env):
    transition_queue = multiprocessing.Queue(100)

    param_queue = multiprocessing.Queue(args.num_actors)
    ps = [multiprocessing.Process(target=leaner_work, args=(args, (transition_queue, param_queue), env))]

    for i in range(args.num_actors):
        ps.append(multiprocessing.Process(target=actor_work, args=(args, (transition_queue, param_queue), env, i)))

    for p in ps:
        p.start()
        time.sleep(0.5)

    for p in ps:
        p.join()


if __name__ == '__main__':
    args = _parse_arg()
    df = dataset.make_dataset()
    columns = df.columns[:-1]
    env = Game(df, columns)
    _train(args, env)