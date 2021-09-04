from gym.envs.registration import register

register(
    id='recon-arena-v0',
    entry_point='gym_marl_reconnaissance.envs.recon_arena:ReconArena',
)
