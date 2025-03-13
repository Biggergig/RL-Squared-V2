from .elo_env import build_env
from .consts import *
import time
import numpy as np
from tqdm import tqdm


def sim_match(elo, model1, model2, n=100, render=False, speed=None):
    env = build_env()

    models = [model1, model2]
    for m in models:
        if m.name not in elo.ratingDict:
            elo.addPlayer(m.name)

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
        if state["result"] == 1:
            elo.gameOver(models[0].name, models[1].name)
        elif state["result"] == 0:
            elo.gameOver(models[0].name, models[1].name, tie=True)
        elif state["result"] == -1:
            elo.gameOver(models[1].name, models[0].name)

    env.close()
    return goals
