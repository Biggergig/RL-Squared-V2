import numpy as np
from rlgym_sim.utils import RewardFunction, math
from rlgym_sim.utils.gamestates import GameState, PlayerData


class DontMoveReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        z = player.car_data.linear_velocity
        return min(-np.log(np.linalg.norm(z)), 10)
