import time
import numpy as np

import rlgym_sim

TPS = 120 / 8

from utils import build_rocketsim_env

env = build_rocketsim_env(name="visgym", phase=1)

while True:
    obs = env.reset()

    done = False
    steps = 0
    ep_reward = 0
    t0 = time.time()
    starttime = time.time()
    while not done:
        actions_1 = env.action_space.sample()
        actions_2 = env.action_space.sample()
        # print(actions_1.reshape((-1, 8)))
        # exit(0)
        actions = np.vstack([actions_1, actions_2])
        # print(actions)
        new_obs, reward, done, state = env.step(actions)
        env.render()
        ep_reward += reward[0]
        steps += 1

        # Sleep to keep the game in real time
        time.sleep(max(0, starttime + steps / TPS - time.time()))

    length = time.time() - t0
    print(
        "Step time: {:1.5f} | Episode time: {:.2f} | Episode Reward: {:.2f}".format(
            length / steps, length, ep_reward
        )
    )
