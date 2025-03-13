from elo_system.sim import sim_match
from elo_system.model import Model
from elo_system.skill import TournamentSkill
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import arguably

model_paths = [os.path.join("data/compare", f) for f in os.listdir("data/compare")]
models = [Model(p) for p in model_paths]

names_to_models = {m.name: m for m in models}

# load all matches from matches.csv
ts = TournamentSkill()
for m in models:
    ts.add_player(m.name)

matches = pd.read_csv("matches.csv", index_col=0)
for _, m1, m2, *goals in tqdm(matches.itertuples(), desc="Loading matches"):
    ts.match(m1, m2, goals)
# print(matches)


def logMatch(name1, name2, goals):
    global matches
    matches.loc[time()] = [name1, name2, *goals]
    # matches.index += 1
    matches.sort_index()
    matches.to_csv("matches.csv", index=True)


# for i in range(len(models)):
#     for j in range(i + 1, len(models)):
#         goals = sim_match(models[i], models[j], 3, render=False)
#         ts.match(models[i].name, models[j].name, goals)
#         logMatch(models[i], models[j], goals)

# print(ts.bots)
# print(ts.getRanks())


def softmax(x, temp=1):
    x = x.copy()
    x /= temp
    return np.exp(x) / sum(np.exp(x))


def chooseTwo(ts, temp=2):
    sig = np.array([t.sigma for t, in ts.bots.values()])
    probs = softmax(sig, temp)
    # print(probs)
    return np.random.choice(list(ts.bots.keys()), 2, p=probs, replace=False)


write_lock = Lock()


def run(args):
    id, render, speed = args
    if id != 0:
        render = False
        speed = None
    print(f"Thread {id} started")
    for _ in range(5):
        # print(ts.getModelsDF(matches).sort_values("name"), "\n")
        if id == 0:
            df = ts.getModelsDF(matches)
            print(df, "TOTAL MATCHES:", df.sum(["win", "draw"]))
        n1, n2 = chooseTwo(ts, temp=0.8)
        # goals = sim_match(names_to_models[n1], names_to_models[n2], 5, render=True, speed=5)
        goals = sim_match(names_to_models[n1], names_to_models[n2], 5, render, speed)
        ts.match(n1, n2, goals)
        with write_lock:
            logMatch(n1, n2, goals)


@arguably.command
def main(*, threads: int = 2, render: bool = False, speed: float = None):
    print("Starting ELO arena with", threads, "threads")
    with ThreadPoolExecutor(threads) as executor:
        try:
            executor.map(run, [(i, render, speed) for i in range(threads)])
        except KeyboardInterrupt:
            print("CTRL-C caught, exiting")
            exit(0)


if __name__ == "__main__":
    arguably.run()
