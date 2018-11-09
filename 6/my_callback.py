from keras.callbacks import Callback
import timeit
import numpy
import warnings
import os

class MyCallback(Callback):
    def __init__(self, output_path="."):
        # Some algorithms compute multiple episodes at once since they are multi-threaded.
        # We therefore use a dictionary that is indexed by the episode to separate episodes
        # from each other.
        self.episode_start = {}
        self.observations = {}
        self.rewards = {}
        self.actions = {}
        self.metrics = {}
        self.step = 0
        self.lastreward = -99999999
        self.output_path = output_path

    def on_train_begin(self, logs):
        self.train_start = timeit.default_timer()
        self.metrics_names = self.model.metrics_names
        print('Training for {} steps ...'.format(self.params['nb_steps']))
        
    def on_train_end(self, logs):
        duration = timeit.default_timer() - self.train_start
        print('done, took {:.3f} seconds'.format(duration))

    def on_episode_begin(self, episode, logs):
        self.episode_start[episode] = timeit.default_timer()
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []
        self.metrics[episode] = []
        

    def on_episode_end(self, episode, logs):
        duration = timeit.default_timer() - self.episode_start[episode]
        episode_steps = len(self.observations[episode])

        # Format all metrics.
        metrics = numpy.array(self.metrics[episode])
        metrics_template = ''
        metrics_variables = []
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            for idx, name in enumerate(self.metrics_names):
                if idx > 0:
                    metrics_template += ', '
                try:
                    value = numpy.nanmean(metrics[:, idx])
                    metrics_template += '{}: {:f}'
                except Warning:
                    value = '--'
                    metrics_template += '{}: {}'
                metrics_variables += [name, value]          
        metrics_text = metrics_template.format(*metrics_variables)

        nb_step_digits = str(int(numpy.ceil(numpy.log10(self.params['nb_steps']))) + 1)
        template = '{step: ' + nb_step_digits + 'd}/{nb_steps}: episode: {episode}, duration: {duration:.3f}s, episode steps: {episode_steps}, steps per second: {sps:.0f}, episode reward: {episode_reward:.3f}, mean reward: {reward_mean:.3f} [{reward_min:.3f}, {reward_max:.3f}], mean action: {action_mean:.3f} [{action_min:.3f}, {action_max:.3f}], {metrics}'
        variables = {
            'step': self.step,
            'nb_steps': self.params['nb_steps'],
            'episode': episode + 1,
            'duration': duration,
            'episode_steps': episode_steps,
            'sps': float(episode_steps) / duration,
            'episode_reward': numpy.sum(self.rewards[episode]),
            'reward_mean': numpy.mean(self.rewards[episode]),
            'reward_min': numpy.min(self.rewards[episode]),
            'reward_max': numpy.max(self.rewards[episode]),
            'action_mean': numpy.mean(self.actions[episode]),
            'action_min': numpy.min(self.actions[episode]),
            'action_max': numpy.max(self.actions[episode]),
            'metrics': metrics_text,
        }
        
        print(template.format(**variables))
        '''
        Code for saving up weights if the episode reward is higher than the last one
        '''
        
        if numpy.sum(self.rewards[episode])>self.lastreward:
            
            previousWeights = "{}/best_weight.hdf5".format(self.output_path)
            if os.path.exists(previousWeights): os.remove(previousWeights)
            self.lastreward = numpy.sum(self.rewards[episode])
            print("The reward is higher than the best one, saving checkpoint weights")
            newWeights = "{}/best_weight.hdf5".format(self.output_path)
            self.model.save_weights(newWeights, overwrite=True)
            
        else:
            print("The reward is lower than the best one, checkpoint weights not updated")
            

        # Free up resources.
        del self.episode_start[episode]
        del self.observations[episode]
        del self.rewards[episode]
        del self.actions[episode]
        del self.metrics[episode]

    def on_step_end(self, step, logs):
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])
        self.metrics[episode].append(logs['metrics'])
        self.step += 1
