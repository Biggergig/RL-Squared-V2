from elo_system.sim import sim_match
from elo_system.model import Model
from elo_system.eloSystem import Elo
import os

model_paths = [os.path.join("data/compare", f) for f in os.listdir("data/compare")]
models = [Model(p) for p in model_paths]

elo = Elo(k=32)

for i in range(len(models)):
    for j in range(i + 1, len(models)):
        sim_match(elo, models[i], models[j], 10)
        print(elo.ratingDict)
