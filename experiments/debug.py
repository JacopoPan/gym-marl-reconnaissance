"""Debugging script.

Create and loop, with a random action, a ReconArena environment.

Example
-------
In a terminal, run as:

    $ python debug.py --gui True --record False --debug True

"""
import os
import time
import argparse
import gym
import yaml
import numpy as np

import gym_marl_reconnaissance

from gym_marl_reconnaissance.envs.recon_arena import ActionType, ObsType, RewardChoice, AdversaryType
from gym_marl_reconnaissance.utils.utils import sync, str2bool
from gym_marl_reconnaissance.utils.greedy import greedy_task_assignment, greedy_tracking


def main():
    """Main function.

    """
    # Set-up parser and parse the script's input arguments.
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gui', default=True, type=str2bool)
    parser.add_argument('--record', default=False, type=str2bool)
    parser.add_argument('--debug', default=True, type=str2bool)
    ARGS = parser.parse_args()
    # Load YAML
    with open(os.path.dirname(os.path.abspath(__file__))+'/defaults.yaml', 'r') as yaml_file:
        YAML_DICT = yaml.safe_load(yaml_file)
    # Merge ARGS and YAML
    YAML_DICT['gui'] = ARGS.gui
    YAML_DICT['record'] = ARGS.record
    YAML_DICT['debug'] = ARGS.debug
    # (TEMP) Dump YAML
    # with open(os.path.dirname(os.path.abspath(__file__))+'/save.yaml', 'w') as outfile:
    #     yaml.dump(YAML_DICT, outfile, default_flow_style=False)
    # Create an environment.
    env = gym.make('recon-arena-v0',
                   **YAML_DICT
                   )
    initial_obs = env.reset()
    START = time.time()
    STEPS = int(1e3)
    num_episodes = 1
    # Step the environment.
    for i in range(STEPS):
        env.render()
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        print(i)
        print(action)
        print(obs)
        print(reward, done)
        print()
        if done:
            _ = env.reset()
            num_episodes += 1
        if ARGS.gui:
            sync(i, START, 1/env.CTRL_FREQ)
    env.close()
    elapsed_sec = time.time() - START
    # Print timing statistics.
    print("\n{:d} control steps (@{:d}Hz) and {:d} episodes in {:.2f} seconds, i.e. {:.2f} steps/sec for a {:.2f}x speedup.\n"
          .format(STEPS, env.CTRL_FREQ, num_episodes, elapsed_sec, STEPS/elapsed_sec, (STEPS/env.CTRL_FREQ)/elapsed_sec))


if __name__ == '__main__':
    main()
