from env import Env
from player import Player
import numpy
import pandas

def load_dataset():
    df = pandas.read_csv("dataset/feature_normalized.csv")
    train_df = df[15800:23200].reset_index(drop=True)
    test_df = df[25000:35000].reset_index(drop=True)
    return train_df, test_df

def main():
    train_df, test_df = load_dataset()
    test_env = Env(test_df)
    action_size, csv_size, img_size = test_env.action_size, test_env.csv_size, test_env.img_size
    player = Player(action_size, csv_size, img_size)
    player.actor.load_weights("model/actor_weight.h5")
    player.critic.load_weights("model/critic_weight.h5")
    test_env.reset()
    test_env.episode_df = test_df
    obs = test_env.observation(test_env.episode_df, test_env.time)
    done = False
    action_list = []
    cap_list = []
    btc_jpy = []
    while not done:        
        csv, img = obs
        action_prob = player.actor.predict([csv, img])[0]
        action = numpy.argmax(action_prob)
        action_list.append(action)
        next_obs, reward, done, info = test_env.step(action)
        obs = next_obs
        cap_list.append(test_env.capability)
        btc_jpy.append(test_env.episode_df["Open"][test_env.time])
        
    df = pandas.DataFrame({"action": action_list, "cap": cap_list, "btc/jpy": btc_jpy})
    df.to_csv("test_df.csv", index=False)
    
if __name__ == "__main__":
    main()
