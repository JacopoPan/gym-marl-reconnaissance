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
import numpy as np

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
    algo = ARGS.exp.split('-')[1]
    # Load the final or best model.
    if os.path.isfile(ARGS.exp+'/success_model.zip'):
        path = ARGS.exp+'/success_model.zip'
    elif os.path.isfile(ARGS.exp+'/best_model.zip'):
        path = ARGS.exp+'/best_model.zip'
    else:
        raise ValueError('[ERROR]: no model under the specified path', ARGS.exp)
    # Create and load the model.
    if algo == 'a2c':
        model = A2C.load(path)
    if algo == 'ppo':
        model = PPO.load(path)
    if algo == 'sac':
        model = SAC.load(path)
    if algo == 'td3':
        model = TD3.load(path)
    if algo == 'ddpg':
        model = DDPG.load(path)
    # Create an evaluation environment.
    eval_env = gym.make('recon-arena-v0')
    # Evaluate the policy.
    mean_reward, std_reward = evaluate_policy(model,
                                              eval_env,
                                              n_eval_episodes=10
                                              )
    print('\n\n\nMean reward ', mean_reward, ' +- ', std_reward, '\n\n')
    # Create a replay environment.
    test_env = gym.make('recon-arena-v0',
                        gui=True,
                        record=True,
                        debug=True
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
