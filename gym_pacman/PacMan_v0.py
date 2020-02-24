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
from gym.envs.classic_control import rendering
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

    Observation is a flattened array of representing the board matrix where
    each cell is either 1 (not visited), 0 (already visited) or 2 (PacMan position).
    See BoardStatus above.

    Action space is [LEFT, RIGHT, UP, DOWN]

    Reward is -1 for each move and 1 for visiting a cell for the first time.
    - Bumping into a wall is -1 (and position won't change)
    - visiting an already visited cell is -1
    - visiting a new cell is -1 + 1 = 0
    Extra bonus reward for clearing the board.
    """

    __version__ = "0.0.1"

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self, config={}):
        """
        Initialise the environment.

        Parameters
        ----------
        All OpenAI-Gym parameters, including:
        config : dict
            board_size: (int, int)      defaults to (5, 5)
            max_moves: int              defaults to {size_x} * {size_y} * 4

        """
        print("PacMan-v0 version: %s" % self.__version__)
        print("numpy version: %s" % np.__version__)
        print("gym version: %s" % gym.__version__)

        # Playing board size
        self._board_size = config.get('board_size', (5, 5))
        self._board = np.full(self._board_size, BoardStatus.DOT)
        logging.debug("Board size: %s", self._board_size)

        # Maximum number of moves
        self._max_moves = config.get('max_moves', self._board_size[0]*self._board_size[1]*4)
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

        # Rendering cell size in px
        self.cell_size = 20

        #  Rendering viewer
        self._viewer = None

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

        # Reset rendering viewer
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

        # Count some stats
        self._episode += 1
        self._nr_moves = 0
        self._total_reward = 0

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

        # Update total reward
        self._total_reward += reward

        # If there are no more dots the episode is over
        if np.count_nonzero(self._board == BoardStatus.DOT) == 0:
            # Board cleared - bonus reward!
            bonus_reward = self._board_size[0]*self._board_size[1]
            self._total_reward += bonus_reward
            reward += bonus_reward

            logging.debug("Board cleared in %s steps. Total reward: %d", self._nr_moves, self._total_reward)
            self.is_over = True

        # If the agent has done 'self._max_moves' and still didn't clear the board the episode is over too
        elif self._nr_moves >= self._max_moves:
            logging.debug("Board not cleared. Total_reward: %d", self._total_reward)
            self.is_over = True

        ret = (self._get_observation(), reward, self.is_over, {})
        return ret

    def _get_cell_value(self, position):
        return self._board[position[0]][position[1]]

    def _set_cell_value(self, position, value):
        self._board[position[0]][position[1]] = value

    def _get_observation(self):
        return self._board.flatten()

    ## Rendering support

    def _idx2geom(self, np_xy):
        # Calculate viewer dot center from numpy array index
        pix_x = np_xy[0] * self.cell_size + self.cell_size/2
        pix_y = np_xy[1] * self.cell_size + self.cell_size/2
        return (pix_x, pix_y)

    def _build_render_board(self):
        board = []
        for x in range(self._board_size[0]):
            board.append([])
            for y in range(self._board_size[1]):
                dot = rendering.make_circle(self.cell_size * 0.5/2)
                dot_pos = self._idx2geom((x, y))
                dot.add_attr(rendering.Transform(translation=dot_pos))
                dot.set_color(0, 0, 0)
                self._viewer.add_geom(dot)
                board[x].append(dot)
        self._render_board = board

    def render(self, mode='human'):
        screen_width = self._board_size[0] * self.cell_size
        screen_height = self._board_size[1] * self.cell_size

        if self._viewer is None:
            self._viewer = rendering.Viewer(screen_width, screen_height)

            self._build_render_board()
            self._pacmantrans = rendering.Transform()
            self._pacman = rendering.make_circle(self.cell_size*0.8/2, filled=True)
            self._pacman.set_color(0.8, 0.8, 0)
            self._pacman.add_attr(self._pacmantrans)
            self._viewer.add_geom(self._pacman)

        # Clear dot under our position
        self._render_board[self.position[0]][self.position[1]].set_color(0.9, 0.9, 0.9)

        # Move PacMan
        new_x, new_y = self._idx2geom(self.position)
        self._pacmantrans.set_translation(new_x, new_y)

        return self._viewer.render(return_rgb_array = mode=='rgb_array')


    def seed(self, seed=None):
        """
        Optionally seed the RNG to get predictable results.

        Parameters
        ----------
        seed : int or None
        """
        random.seed(seed)
        np.random.seed(seed)


    def close(self):
        if self._viewer:
            self._viewer.close()
            self._viewer = None
