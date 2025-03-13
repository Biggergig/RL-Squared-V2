from elo_system.sim import sim_match
from elo_system.model import Model
from elo_system.eloSystem import Elo

model_1_path = "data/checkpoints/curriculum_tiny/remote/curriculum_tiny/1/100028602"
model_2_path = "data/checkpoints/curriculum_tiny/remote/curriculum_tiny/1/123835954"


elo = Elo(k=32)

goals = sim_match(
    elo, Model(model_1_path, "400m"), Model(model_2_path, "100m"), 100, render=True
)
print(elo.ratingDict)
# goals = sim_match(elo, Model(model_1_path, "400m"), Model(model_1_path, "400m"), 10)
# print(elo.ratingDict)
