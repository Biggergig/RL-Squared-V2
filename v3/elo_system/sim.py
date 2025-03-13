from .elo_env import build_env
from .consts import *
import time
import numpy as np
from tqdm import tqdm


def sim_match(elo, model1, model2, n=100, render=False, speed=None):
    env = build_env()

    models = [model1, model2]

    goals = [0, 0, 0]
    # orange, timeout, blue
    for _ in (pbar := tqdm(range(n))):
        pbar.set_description(f"Goals: {goals}")
        obs = env.reset()

        done = False
        steps = 0
        t0 = time.time()
        starttime = time.time()
        while not done:
            # if steps % tick_skip == 0:
            actions = np.vstack([m.act(obs[i]) for i, m in enumerate(models)])
            obs, reward, done, state = env.step(actions)
            steps += 1
            if render:
                env.render()
                if speed is not None:
                    time.sleep(max(0, starttime + steps / (TPS * speed) - time.time()))
        goals[-int(state["result"]) + 1] += 1
        length = time.time() - t0
        # print(
        #     "Step time: {:1.5f} | Episode time: {:.2f} | Goals: {}".format(
        #         length / steps, length, goals
        #     )
        # )
    env.close()
    return goals
