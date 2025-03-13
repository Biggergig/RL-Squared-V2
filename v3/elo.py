from elo_system.sim import sim_match
from elo_system.model import Model
from elo_system.skill import TournamentSkill
import os
import pandas as pd

model_paths = [os.path.join("data/compare", f) for f in os.listdir("data/compare")]
models = [Model(p) for p in model_paths]

# load all matches from matches.csv
ts = TournamentSkill()
for m in models:
    ts.add_player(m.name)

matches = pd.read_csv("matches.csv")
for _, m1, m2, *goals in matches.itertuples():
    ts.match(m1, m2, goals)


def logMatch(model1, model2, goals):
    global matches
    matches.loc[-1] = [model1.name, model2.name, *goals]
    matches.index += 1
    matches.sort_index()
    matches.to_csv("matches.csv", index=False)


# for i in range(len(models)):
#     for j in range(i + 1, len(models)):
#         goals = sim_match(models[i], models[j], 3, render=False)
#         ts.match(models[i].name, models[j].name, goals)
#         logMatch(models[i], models[j], goals)

print(ts.bots)
# print(*ts.getElos(), sep="\n")
# print(ts.model.predict_rank(list(ts.bots.values())))
print(ts.getRanks())
