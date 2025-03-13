from elo_system.sim import sim_match
from elo_system.model import Model
from elo_system.skill import TournamentSkill
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time
from multiprocessing import Pool, Lock
import arguably

model_paths = [
    os.path.join("elo_system/compare", f)
    for f in os.listdir("elo_system/compare")
    if f != ".gitignore"
]
models = [Model(p) for p in model_paths]

names_to_models = {m.name: m for m in models}

# load all matches from matches.csv
ts = TournamentSkill()
for m in models:
    ts.add_player(m.name)

matches = pd.read_csv("matches.csv", index_col=0)
for _, m1, m2, *goals in tqdm(matches.itertuples(), desc="Loading matches"):
    ts.match(m1, m2, goals)


def logMatch(name1, name2, goals):
    global matches
    matches = pd.read_csv("matches.csv", index_col=0)
    matches.loc[time()] = [name1, name2, *goals]
    matches.sort_index()
    matches.to_csv("matches.csv", index=True)


def softmax(x, temp=1):
    x = x.copy()
    x /= temp
    return np.exp(x) / sum(np.exp(x))


def init_child(lock_):
    global lock
    lock = lock_


def run(model1, model2, speed):
    goals = sim_match(model1, model2, 15, True, speed)


@arguably.command
def main(
    name1: str,
    name2: str,
    *,
    speed: float = None,
):
    run(names_to_models[name1], names_to_models[name2], speed)


if __name__ == "__main__":
    arguably.run()
