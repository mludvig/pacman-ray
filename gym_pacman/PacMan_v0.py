#!/usr/bin/env python3

"""
PacMan simulator v0

This environment returns 'observation' as a flattened numpy array of BoardStatus values.
The PacMan position is shown on the array (cell with BoardStatus.PACMAN value).

By Michael Ludvig
"""

import logging
import random
from enum import IntEnum

import gym
from gym import spaces
import numpy as np

logging.basicConfig(format='%(levelname)s %(message)s', level=logging.DEBUG)


class BoardStatus(IntEnum):
    EMPTY = 0
    DOT = 1
    PACMAN = 2


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class PacMan_v0(gym.Env):
    """
    The environment defines which actions can be taken at which point and
    when the agent receives which reward.

    Observation is an array of:
    - 10x10 matrix where each cell is either 1 (not visited) or 0 (already visited)
    - pacman position on the board as (x, y) tuple

    Action space is [LEFT, RIGHT, UP, DOWN]

    Reward is -1 for each move and 1 for visiting a cell for the first time.
    - Bumping into a wall is -1 (and position won't change)
    - visiting an already visited cell is -1
    - visiting a new cell is -1 + 1 = 0
    """

    __version__ = "0.0.1"

    def __init__(self, additional_simulator_parameters={}):
        """
        Initialise the environment.

        Parameters
        ----------
        All OpenAI-Gym parameters, including:
        additional_simulator_parameters : dict
            board_size: (int, int)      defaults to (10, 10)
            max_moves: int              defaults to 1000

        """
        print("PacMan-v0 version: %s" % self.__version__)
        print("numpy version: %s" % np.__version__)
        print("gym version: %s" % gym.__version__)

        # Playing board size
        self._board_size = additional_simulator_parameters.get('board_size', (5, 5))
        self._board = np.full(self._board_size, BoardStatus.DOT)
        logging.debug("Board size: %s", self._board_size)

        # Maximum number of moves
        self._max_moves = additional_simulator_parameters.get('max_moves', 500)
        logging.debug("Max moves: %s", self._max_moves)

        # The actions the agent can choose from (must be named 'self.action_space')
        self.action_space = spaces.Discrete(max(Action) + 1)

        # Observation is what we return back to the agent
        self.observation_space = spaces.Box(
               low=np.full(self._board_size, min(BoardStatus)).flatten(),
               high=np.full(self._board_size, max(BoardStatus)).flatten(),
               dtype=np.int32)

        # Episode counter
        self._episode = 0

        self.reset()

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """

        # Random pacman position
        self.position = np.array([
            np.random.randint(self._board_size[0]),
            np.random.randint(self._board_size[1]),
        ])

        # Initialise the board
        self._board = np.full(self._board_size, BoardStatus.DOT)
        self._set_cell_value(self.position, BoardStatus.PACMAN)

        # Episode is not yet over
        self.is_over = False

        # Count some stats
        self._episode += 1
        self._nr_moves = 0

        return self._get_observation()

    def step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : Action() [int]

        Returns
        -------
        observation, reward, episode_over, info : tuple
        """

        if self.is_over:
            raise RuntimeError("Episode is done")

        # Each step costs 1 reward point
        reward = -1

        # Increment moves
        self._nr_moves += 1

        # Clear cell before updating position
        self._set_cell_value(self.position, BoardStatus.EMPTY)

        # Is it a valid move?
        if action == Action.UP and self.position[0] > 0:
            self.position[0] -= 1
        elif action == Action.DOWN and self.position[0] < self._board_size[0] - 1:
            self.position[0] += 1
        elif action == Action.LEFT and self.position[1] > 0:
            self.position[1] -= 1
        elif action == Action.RIGHT and self.position[1] < self._board_size[1] - 1:
            self.position[1] += 1
        # else we don't change position

        # Move PacMan to the new position
        if self._get_cell_value(self.position) == BoardStatus.DOT:
            reward += 1
            #self._set_cell_value(self.position, BoardStatus.EMPTY)
        self._set_cell_value(self.position, BoardStatus.PACMAN)

        # If there are no more dots the episode is over
        if np.count_nonzero(self._board == BoardStatus.DOT) == 0:
            logging.debug("Board cleared in %s steps", self._nr_moves)
            self.is_over = True

        # If the agent has done 1000+ moves and still didn't clear the board the episode is over too
        elif self._nr_moves >= self._max_moves:
            print("Board not cleared")
            self.is_over = True

        ret = (self._get_observation(), reward, self.is_over, {})
        return ret

    def _get_cell_value(self, position):
        return self._board[position[0]][position[1]]

    def _set_cell_value(self, position, value):
        self._board[position[0]][position[1]] = value

    def _get_observation(self):
        return self._board.flatten()

    def render(self, mode='human'):
        return

    def seed(self, seed=None):
        """
        Optionally seed the RNG to get predictable results.

        Parameters
        ----------
        seed : int or None
        """
        random.seed(seed)
        np.random.seed(seed)
