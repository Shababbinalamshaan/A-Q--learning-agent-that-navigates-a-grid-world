"""
train.py

Trains a Q-learning agent on the GridWorld environment, then:
  1. Prints the learned optimal policy as a grid of arrows.
  2. Saves a plot of total reward per episode to learning_curve.png,
     showing the agent improving over time.

Run with:
    python train.py
"""

import matplotlib
matplotlib.use("Agg")  # write plots to file, no display needed
import matplotlib.pyplot as plt

from gridworld import GridWorld
from q_learning_agent import QLearningAgent

NUM_EPISODES = 500
MAX_STEPS_PER_EPISODE = 100

ARROW = {"UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→"}


def train():
    env = GridWorld(
        size=5,
        pits=[(1, 1), (2, 3), (3, 1)],
        start=(0, 0),
        goal=(4, 4),
    )
    agent = QLearningAgent(actions=GridWorld.ACTIONS)

    episode_rewards = []

    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0.0

        for _ in range(MAX_STEPS_PER_EPISODE):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

    return env, agent, episode_rewards


def print_policy(env, agent):
    """Print the greedy action learned for every walkable cell."""
    print("Learned policy (S = start, G = goal, X = pit):\n")
    for r in range(env.size):
        row_str = ""
        for c in range(env.size):
            cell = (r, c)
            if cell == env.goal:
                row_str += "G  "
            elif cell in env.pits:
                row_str += "X  "
            elif cell == env.start:
                row_str += ARROW[agent.best_action(cell)] + "  "
            else:
                row_str += ARROW[agent.best_action(cell)] + "  "
        print(row_str)
    print()


def plot_learning_curve(episode_rewards, filename="learning_curve.png"):
    """Save a plot of reward-per-episode, with a rolling average."""
    window = 20
    rolling_avg = [
        sum(episode_rewards[max(0, i - window):i + 1]) / len(episode_rewards[max(0, i - window):i + 1])
        for i in range(len(episode_rewards))
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(episode_rewards, alpha=0.3, label="Episode reward")
    plt.plot(rolling_avg, color="red", linewidth=2, label=f"{window}-episode rolling average")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Q-Learning on GridWorld: Reward per Episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved learning curve to {filename}")


if __name__ == "__main__":
    env, agent, episode_rewards = train()
    print_policy(env, agent)
    plot_learning_curve(episode_rewards)

    print(f"Average reward, first 20 episodes:  {sum(episode_rewards[:20]) / 20:.2f}")
    print(f"Average reward, last 20 episodes:   {sum(episode_rewards[-20:]) / 20:.2f}")
