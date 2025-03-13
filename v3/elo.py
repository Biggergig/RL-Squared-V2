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


choices = [
    # ("child_654m", "curriculum_306m"),
    # ("child_654m", "curriculum_wide_60m"),
    # ("child_654m", "gentle_tiny_30m"),
    # ("child_654m", "curriculum_wide_30m"),
    # ("child_654m", "gentle_bigcrit_38m"),
    # ("child_654m", "curriculum_tiny_30m"),
    # ("grounded_100m", "gentle_260m"),
    # ("grounded_100m", "curriculum_tiny_100m"),
    # ("grounded_100m", "curriculum_wide_114m"),
    # ("grounded_100m", "curriculum_wide_60m"),
    # ("grounded_100m", "curriculum_61m"),
    # ("grounded_100m", "gentle_tiny_60m"),
    # ("grounded_100m", "curriculum_wide_30m"),
    # ("curriculum_405m", "bulldog_470m"),
    # ("curriculum_405m", "curriculum_204m"),
    # ("curriculum_405m", "grounded_30m"),
    # ("curriculum_405m", "curriculum_wide_114m"),
    # ("curriculum_405m", "gentle_tiny_100m"),
    # ("grounded_60m", "curriculum_slow_100m"),
    # ("grounded_60m", "curriculum_wide_60m"),
    # ("grounded_60m", "gentle_tiny_100m"),
    # ("grounded_60m", "curriculum_30m"),
    # ("grounded_60m", "sparse_60m"),
    # ("gentle_105m", "gentle_63m"),
    # ("gentle_105m", "gentle_bigcrit_66m"),
    # ("gentle_105m", "curriculum_wide_114m"),
    # ("gentle_105m", "curriculum_slow_100m"),
    # ("gentle_105m", "sparse_60m"),
    # ("gentle_105m", "gentle_33m"),
    # ("gentle_105m", "curriculum_wide_30m"),
    # ("gentle_105m", "gentle_bigcrit_38m"),
    # ("bulldog_470m", "curriculum_tiny_100m"),
    # ("bulldog_470m", "curriculum_wide_114m"),
    # ("bulldog_470m", "curriculum_102m"),
    # ("bulldog_470m", "curriculum_slow_100m"),
    # ("bulldog_470m", "curriculum_wide_60m"),
    # ("bulldog_470m", "deeper_gentle_128m"),
    # ("bulldog_470m", "curriculum_tiny_60m"),
    # ("bulldog_470m", "pure_ppo_780m"),
    # ("gentle_63m", "curriculum_306m"),
    # ("gentle_63m", "grounded_30m"),
    # ("gentle_63m", "curriculum_slow_100m"),
    # ("gentle_63m", "curriculum_wide_60m"),
    # ("gentle_63m", "gentle_tiny_30m"),
    # ("gentle_63m", "gentle_tiny_60m"),
    # ("curriculum_306m", "gentle_bigcrit_66m"),
    # ("curriculum_306m", "curriculum_wide_60m"),
    # ("curriculum_306m", "gentle_tiny_60m"),
    # ("curriculum_306m", "gentle_bigcrit_104m"),
    # ("gentle_260m", "curriculum_wide_114m"),
    # ("gentle_260m", "curriculum_wide_60m"),
    # ("gentle_260m", "gentle_tiny_100m"),
    # ("gentle_260m", "curriculum_61m"),
    # ("gentle_260m", "gentle_tiny_30m"),
    # ("gentle_260m", "gentle_bigcrit_38m"),
    # ("curriculum_204m", "curriculum_tiny_100m"),
    # ("curriculum_204m", "curriculum_wide_114m"),
    # ("curriculum_204m", "curriculum_tiny_60m"),
    # ("curriculum_204m", "default_deep_136m"),
    # ("curriculum_204m", "curriculum_wide_30m"),
    # ("curriculum_204m", "gentle_bigcrit_38m"),
    # ("curriculum_204m", "curriculum_tiny_30m"),
    # ("grounded_30m", "curriculum_wide_114m"),
    # ("grounded_30m", "curriculum_wide_60m"),
    # ("grounded_30m", "gentle_tiny_60m"),
    # ("gentle_bigcrit_66m", "curriculum_wide_114m"),
    # ("gentle_bigcrit_66m", "gentle_tiny_100m"),
    # ("gentle_bigcrit_66m", "curriculum_tiny_60m"),
    # ("gentle_bigcrit_66m", "curriculum_wide_30m"),
    # ("gentle_bigcrit_66m", "pure_ppo_780m"),
    # ("curriculum_tiny_100m", "curriculum_wide_60m"),
    # ("curriculum_tiny_100m", "curriculum_30m"),
    # ("curriculum_tiny_100m", "gentle_tiny_30m"),
    # ("curriculum_tiny_100m", "curriculum_tiny_60m"),
    # ("curriculum_wide_114m", "curriculum_slow_100m"),
    # ("curriculum_wide_114m", "deeper_gentle_128m"),
    # ("curriculum_wide_114m", "curriculum_tiny_60m"),
    # ("curriculum_wide_114m", "gentle_bigcrit_104m"),
    # ("curriculum_wide_114m", "default_deep_136m"),
    # ("curriculum_wide_114m", "gentle_bigcrit_38m"),
    # ("curriculum_wide_114m", "curriculum_tiny_30m"),
    # ("curriculum_102m", "curriculum_wide_60m"),
    # ("curriculum_102m", "sparse_60m"),
    # ("curriculum_102m", "gentle_tiny_30m"),
    ("curriculum_102m", "gentle_tiny_60m"),
    ("curriculum_102m", "curriculum_wide_30m"),
    ("curriculum_slow_100m", "curriculum_30m"),
    ("curriculum_slow_100m", "curriculum_61m"),
    ("curriculum_slow_100m", "curriculum_wide_30m"),
    ("curriculum_slow_100m", "curriculum_tiny_30m"),
    ("curriculum_wide_60m", "gentle_tiny_100m"),
    ("curriculum_wide_60m", "gentle_tiny_30m"),
    ("curriculum_wide_60m", "curriculum_tiny_60m"),
    ("curriculum_wide_60m", "gentle_33m"),
    ("curriculum_wide_60m", "curriculum_tiny_30m"),
    ("curriculum_wide_60m", "pure_ppo_780m"),
    ("gentle_tiny_100m", "sparse_60m"),
    ("gentle_tiny_100m", "gentle_33m"),
    ("gentle_tiny_100m", "curriculum_wide_30m"),
    ("curriculum_30m", "deeper_gentle_128m"),
    ("curriculum_30m", "gentle_tiny_60m"),
    ("deeper_gentle_128m", "curriculum_61m"),
    ("deeper_gentle_128m", "curriculum_tiny_60m"),
    ("deeper_gentle_128m", "curriculum_wide_30m"),
    ("curriculum_61m", "sparse_60m"),
    ("curriculum_61m", "gentle_tiny_30m"),
    ("curriculum_61m", "default_deep_136m"),
    ("curriculum_61m", "pure_ppo_780m"),
    ("sparse_60m", "gentle_tiny_60m"),
    ("sparse_60m", "gentle_33m"),
    ("sparse_60m", "curriculum_wide_30m"),
    ("gentle_tiny_30m", "curriculum_wide_30m"),
    ("gentle_tiny_60m", "pure_ppo_780m"),
    ("curriculum_tiny_60m", "curriculum_tiny_30m"),
    ("curriculum_tiny_60m", "pure_ppo_780m"),
    ("gentle_bigcrit_104m", "gentle_bigcrit_38m"),
    ("gentle_33m", "curriculum_wide_30m"),
    ("gentle_33m", "gentle_bigcrit_38m"),
    ("gentle_33m", "curriculum_tiny_30m"),
    ("default_deep_136m", "pure_ppo_780m"),
    ("gentle_bigcrit_38m", "curriculum_tiny_30m"),
    ("curriculum_tiny_30m", "pure_ppo_780m"),
]


def chooseTwo(ts, top_k=-1, temp=2):
    return choices.pop(0)
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
    threads: int = 2,
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
