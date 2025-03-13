from elo_system.sim import sim_match
from elo_system.model import Model
from elo_system.eloSystem import Elo

model_1_path = "data/checkpoints/curriculum/remote/curriculum/1/405360906"
model_2_path = "data/checkpoints/curriculum/remote/curriculum/1/102016886"

elo = Elo(k=32)

goals = sim_match(elo, Model(model_1_path, "400m"), Model(model_2_path, "100m"), 10)
for n in range(goals[0]):
    elo.gameOver("400m", "100m")
for n in range(goals[1]):
    elo.gameOver("400m", "100m", tie=True)
for n in range(goals[2]):
    elo.gameOver("100m", "400m")

print(elo.ratingDict)
