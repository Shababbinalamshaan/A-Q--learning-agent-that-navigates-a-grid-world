"""
q_learning_agent.py

A tabular Q-learning agent.

Q-learning learns a table Q(state, action) that estimates the expected
total future reward of taking `action` in `state` and then acting
optimally afterwards. It updates this table using the Bellman equation:

    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

where:
    alpha  = learning rate (how much each new experience updates the table)
    gamma  = discount factor (how much future reward matters vs. immediate)
    r      = reward received after taking action a in state s
    s'     = next state reached
"""

import random
import numpy as np


class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        """
        Args:
            actions: list of possible actions, e.g. ["UP","DOWN","LEFT","RIGHT"]
            alpha: learning rate
            gamma: discount factor for future rewards
            epsilon: initial exploration rate (probability of a random action)
            epsilon_decay: multiply epsilon by this after every episode
            epsilon_min: floor so the agent never stops exploring completely
        """
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}  # maps state -> np.array of Q-values per action

    def _get_q_values(self, state):
        """Return the Q-value row for a state, creating it if unseen."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        return self.q_table[state]

    def choose_action(self, state):
        """Epsilon-greedy action selection: explore vs. exploit."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)  # explore
        q_values = self._get_q_values(state)
        best_index = int(np.argmax(q_values))
        return self.actions[best_index]  # exploit

    def update(self, state, action, reward, next_state, done):
        """Apply one Q-learning update after observing a transition."""
        action_index = self.actions.index(action)
        q_values = self._get_q_values(state)
        next_q_values = self._get_q_values(next_state)

        target = reward if done else reward + self.gamma * np.max(next_q_values)
        q_values[action_index] += self.alpha * (target - q_values[action_index])

    def decay_epsilon(self):
        """Reduce exploration rate after each episode, down to a floor."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def best_action(self, state):
        """Return the greedy (no exploration) action for a state."""
        q_values = self._get_q_values(state)
        return self.actions[int(np.argmax(q_values))]
