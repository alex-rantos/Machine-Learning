# https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
from time import sleep
from IPython.display import clear_output
import gym
import random
import numpy as np


def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)


def train(env):
    # Hyperparameters
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1

    # For plotting metrics
    all_epochs = []
    all_penalties = []

    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    for i in range(1, 100001):
        state = env.reset()

        epochs, penalties, reward, = 0, 0, 0
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(q_table[state])  # Exploit learned values

            next_state, reward, done, info = env.step(action)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * \
                (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1

        if i % 100 == 0:
            clear_output(wait=True)
            print(f"Episode: {i}")

    print("Training finished.\n")
    return q_table


def eval(env, q_table):
    """Evaluate agent's performance after Q-learning"""

    total_epochs, total_penalties = 0, 0
    episodes = 100

    for _ in range(episodes):
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0

        done = False

        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, info = env.step(action)

            if reward == -10:
                penalties += 1

            epochs += 1

        total_penalties += penalties
        total_epochs += epochs

    print(f"Results after {episodes} episodes:")
    print(f"Average timesteps per episode: {total_epochs / episodes}")
    print(f"Average penalties per episode: {total_penalties / episodes}")


if __name__ == "__main__":

    env = gym.make("Taxi-v3").env

    env.reset()

    print("Action Space {}".format(env.action_space))
    print("State Space {}".format(env.observation_space))

    # (taxi row, taxi column, passenger index, destination index)
    state = env.encode(3, 1, 2, 0)
    print("State:", state)

    env.s = state
    env.render()

    print(env.P[399])
    env.s = 328  # set environment to illustration's state

    epochs = 0
    penalties, reward = 0, 0

    frames = []  # for animation

    done = False

    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        # Put each rendered frame into dict for animation
        frames.append({
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'reward': reward
        }
        )

        epochs += 1

    print("Timesteps taken: {}".format(epochs))
    print("Penalties incurred: {}".format(penalties))

    # print_frames(frames)
    q_table = train(env)
    print(q_table)
    eval(env, q_table)
