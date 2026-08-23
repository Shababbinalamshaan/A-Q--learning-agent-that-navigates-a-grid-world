"""
gridworld.py

A simple 2D grid-world environment for reinforcement learning.

The agent starts at the top-left corner and must reach the goal at the
bottom-right corner, while avoiding "pit" cells that give a large negative
reward. This is a classic toy environment used to demonstrate tabular
Q-learning.

Grid legend (for render()):
    A = agent's current position
    G = goal
    X = pit (bad terminal state)
    . = empty, walkable cell
"""

import numpy as np


class GridWorld:
    # Four possible actions the agent can take
    ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
    ACTION_DELTAS = {
        "UP": (-1, 0),
        "DOWN": (1, 0),
        "LEFT": (0, -1),
        "RIGHT": (0, 1),
    }

    def __init__(self, size=5, pits=None, start=(0, 0), goal=None):
        """
        Args:
            size: the grid is size x size
            pits: list of (row, col) tuples that end the episode with a
                  large negative reward if the agent steps on them
            start: (row, col) starting cell for every episode
            goal: (row, col) goal cell; defaults to bottom-right corner
        """
        self.size = size
        self.start = start
        self.goal = goal if goal is not None else (size - 1, size - 1)
        self.pits = pits if pits is not None else []
        self.agent_pos = start

    def reset(self):
        """Move the agent back to the start of a new episode."""
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action):
        """
        Apply an action and return (next_state, reward, done).

        Reward structure:
            -1     for every normal step (encourages shorter paths)
            +10    for reaching the goal
            -10    for falling into a pit
        """
        dr, dc = self.ACTION_DELTAS[action]
        r, c = self.agent_pos
        new_r = min(max(r + dr, 0), self.size - 1)
        new_c = min(max(c + dc, 0), self.size - 1)
        self.agent_pos = (new_r, new_c)

        if self.agent_pos == self.goal:
            return self.agent_pos, 10.0, True
        if self.agent_pos in self.pits:
            return self.agent_pos, -10.0, True
        return self.agent_pos, -1.0, False

    def render(self):
        """Print a text picture of the current grid."""
        for r in range(self.size):
            row_str = ""
            for c in range(self.size):
                cell = (r, c)
                if cell == self.agent_pos:
                    row_str += "A "
                elif cell == self.goal:
                    row_str += "G "
                elif cell in self.pits:
                    row_str += "X "
                else:
                    row_str += ". "
            print(row_str)
        print()
