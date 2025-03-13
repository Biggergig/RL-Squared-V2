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


def chooseTwo(ts, top_k=-1, temp=2):
    return ["bulldog_470m"] * 2
    return "bulldog_470m", "child_654m"
    if top_k != -1:
        top_bots = ts.getModelsDF(matches).sort_values("rank").head(top_k).index
        # print("loading from", top_bots)
        return np.random.choice(list(top_bots), 2, replace=False)
    sig = np.array([t.sigma for t, in ts.bots.values()])
    probs = softmax(sig, temp)
    # print(probs)
    return np.random.choice(list(ts.bots.keys()), 2, p=probs, replace=False)


def init_child(lock_):
    global lock
    lock = lock_


def run(args):
    id, batch_size, iters, top_k, render, speed = args
    if id != 0:
        render = False
        speed = None
    print(f"Thread {id} started")
    for _ in range(iters):
        batch = []
        for _ in range(batch_size):
            # print(ts.getModelsDF(matches).sort_values("name"), "\n")

            n1, n2 = chooseTwo(ts, top_k=top_k, temp=0.8)
            # goals = sim_match(names_to_models[n1], names_to_models[n2], 5, render=True, speed=5)
            goals = sim_match(
                names_to_models[n1], names_to_models[n2], 15, render, speed
            )
            ts.match(n1, n2, goals)
            batch.append([n1, n2, goals])
        with lock:
            for n1, n2, goals in batch:
                print(f"{id}: writing", n1, n2, goals)
                logMatch(n1, n2, goals)


@arguably.command
def main(
    *,
    threads: int = 1,
    batch_size: int = 2,
    iters: int = 1,
    render: bool = False,
    speed: float = None,
    top_k: int = -1,
):
    if threads > 0:
        print("Starting ELO arena with", threads, "threads")
        write_lock = Lock()
        with Pool(threads, initializer=init_child, initargs=(write_lock,)) as pool:
            pool.map(
                run,
                [(i, batch_size, iters, top_k, render, speed) for i in range(threads)],
            )
    df = ts.getModelsDF(matches).sort_values("name")
    print(
        df,
        df.sort_values("rank"),
        f"TOTAL GOALS: {int( df.win.sum() + df.draw.sum() / 2 )}",
        sep="\n\n",
    )


if __name__ == "__main__":
    arguably.run()
