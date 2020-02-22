import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='PacMan-v0',
    entry_point='gym_pacman.PacMan_v0:PacMan_v0',
)

register(
    id='PacMan-v1',
    entry_point='gym_pacman.PacMan_v1:PacMan_v1',
)
