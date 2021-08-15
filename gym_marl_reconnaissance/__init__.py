from gym.envs.registration import register

register(
    id='env-v0',
    entry_point='gym_marl_reconnaissance.envs:EnvClassName',
)
