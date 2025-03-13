from elo_system.sim import sim_match
from elo_system.model import Model
from elo_system.skill import TournamentSkill
import os
import pandas as pd

model_paths = [os.path.join("data/compare", f) for f in os.listdir("data/compare")][-3:]
models = [Model(p) for p in model_paths]

ts = TournamentSkill()

pd.DataFrame([], columns=["Model1", "Model2", "Orange", "Tie", "Blue"]).to_csv(
    "matches.csv"
)


# for i in range(len(models)):
#     for j in range(i + 1, len(models)):
#         goals = sim_match(models[i], models[j], 3, render=True)
#         ts.match(models[i].name, models[j].name, goals)

print(ts.bots)
print(ts.ranks())
