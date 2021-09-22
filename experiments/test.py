"""Testing script.

Create a ReconArena environment and re-play a trained policy.

Example
-------
In a terminal, run as:

    $ python test.py --exp ../results/exp-<algo>-<date>_<time>

"""
import os
import time
import argparse
import subprocess
import gym
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy

from gym_marl_reconnaissance.envs.recon_arena import ActionType, ObsType, RewardChoice, AdversaryType, ReconArena
from gym_marl_reconnaissance.utils.utils import sync, str2bool


def main():
    """Main function.

    """
    # Set-up parser and parse the script's input arguments.
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--exp', type=str)
    ARGS = parser.parse_args()
    # Load YAML.
    with open(ARGS.exp+'/save.yaml', 'r') as yaml_file:
        YAML_DICT = yaml.safe_load(yaml_file)
    # Load the final or best model.
    if os.path.isfile(ARGS.exp+'/success_model.zip'):
        path = ARGS.exp+'/success_model.zip'
    elif os.path.isfile(ARGS.exp+'/best_model.zip'):
        path = ARGS.exp+'/best_model.zip'
    else:
        raise ValueError('[ERROR] no model under the specified path', ARGS.exp)
    # Create and load the model.
    if YAML_DICT['algo'] == 'a2c':
        model = A2C.load(path)
    if YAML_DICT['algo'] == 'ppo':
        model = PPO.load(path)
    if YAML_DICT['algo'] == 'sac':
        model = SAC.load(path)
    if YAML_DICT['algo'] == 'td3':
        model = TD3.load(path)
    if YAML_DICT['algo'] == 'ddpg':
        model = DDPG.load(path)
    YAML_DICT.pop('algo')
    # Plot evaluations
    with np.load(ARGS.exp+'/evaluations.npz') as data:
        # print(data.files)
        # print(data['timesteps'])
        # print(data['results'])
        # print(data['ep_lengths'])
        print('mean', np.mean(data['results']), 'std', np.std(data['results']),
              'max', np.max(data['results']), 'min', np.min(data['results']),
              'Q1', np.percentile(data['results'], 25), 'median', np.percentile(data['results'], 50), 'Q3', np.percentile(data['results'], 75)
              )
        length = len(data['results'])
        print('last 90 percent of training')
        start_index = length//10
        remaining_data = data['results'][start_index:]
        print('mean', np.mean(remaining_data), 'std', np.std(remaining_data),
              'max', np.max(remaining_data), 'min', np.min(remaining_data),
              'Q1', np.percentile(remaining_data, 25), 'median', np.percentile(remaining_data, 50), 'Q3', np.percentile(remaining_data, 75)
              )
        print('last 80 percent of training')
        start_index = length//5
        remaining_data = data['results'][start_index:]
        print('mean', np.mean(remaining_data), 'std', np.std(remaining_data),
              'max', np.max(remaining_data), 'min', np.min(remaining_data),
              'Q1', np.percentile(remaining_data, 25), 'median', np.percentile(remaining_data, 50), 'Q3', np.percentile(remaining_data, 75)
              )
        fig, (ax0) = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(12, 6))
        ax0.set_title('all errorbars')
        ax0.errorbar(data['timesteps'], np.mean(data['results'], axis=1), yerr=np.std(data['results'], axis=1))
        # ax0.errorbar(data['timesteps'], np.mean(data['results'], axis=1)) #, yerr=y2err)
        fig.suptitle('Errorbar subsampling')
        plt.show()
        exit()
    # Create an evaluation environment.
    eval_env = gym.make('recon-arena-v0', **YAML_DICT)
    # Evaluate the policy.
    mean_reward, std_reward = evaluate_policy(model,
                                              eval_env,
                                              n_eval_episodes=10
                                              )
    print('\n\n\nMean reward ', mean_reward, ' +- ', std_reward, '\n\n')
    # Create a replay environment.
    YAML_DICT['gui'] = True
    YAML_DICT['record'] = True
    YAML_DICT['debug'] = True
    test_env = gym.make('recon-arena-v0',
                        **YAML_DICT
                        )
    # Replay the trained model.
    obs = test_env.reset()
    START = time.time()
    for i in range(int(test_env.EPISODE_LENGTH_SEC * test_env.CTRL_FREQ)):
        action, _ = model.predict(obs,
                                  deterministic=True # deterministic=False
                                  )
        obs, reward, done, info = test_env.step(action)
        print(action)
        print(obs)
        print(reward)
        print()
        test_env.render()
        sync(i, START, 1/test_env.CTRL_FREQ)
    test_env.close()


if __name__ == '__main__':
    main()
