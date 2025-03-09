import numpy as np

from rlgym_sim.utils import RewardFunction, math
from rlgym_sim.utils.common_values import BALL_RADIUS, CAR_MAX_SPEED
from rlgym_sim.utils.gamestates import GameState, PlayerData


class DistancePlayerToBallReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        # Compensate for inside of ball being unreachable (keep max reward at 1)
        dist = (
            np.linalg.norm(player.car_data.position - state.ball.position) - BALL_RADIUS
        )
        return dist
