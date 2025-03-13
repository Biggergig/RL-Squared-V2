from elo_system.sim import sim_match
from elo_system.model import Model
from elo_system.eloSystem import Elo

model_1_path = "data/checkpoints/curriculum/remote/curriculum/1/405360906"
model_2_path = "data/checkpoints/curriculum/remote/curriculum/1/102016886"

elo = Elo(k=32)

goals = sim_match(elo, Model(model_1_path, "400m"), Model(model_2_path, "100m"), 10)
print(elo.ratingDict)
goals = sim_match(elo, Model(model_1_path, "400m"), Model(model_1_path, "400m"), 10)
print(elo.ratingDict)
