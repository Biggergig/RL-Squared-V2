import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData


class ClippedReward(RewardFunction):  # We extend the class "RewardFunction"
    # Empty default constructor (required)
    def __init__(
        self,
        reward_fn: RewardFunction,
        lb=float("-inf"),
        ub=float("inf"),
    ):
        super().__init__()
        self.fn = reward_fn
        self.lb = lb
        self.ub = ub

    # Called when the game resets (i.e. after a goal is scored)
    def reset(self, initial_state: GameState):
        self.fn.reset(initial_state)

    # Get the reward for a specific player, at the current state
    def get_reward(
        self, player: PlayerData, state: GameState, previous_action
    ) -> float:
        rew = self.fn.get_reward(player, state, previous_action)
        return max(lb, min(ub, rew))

    # Get the reward for a specific player, at the current state
    def get_final_reward(self, *args, **kwargs) -> float:
        rew = self.fn.get_final_reward(*args, **kwargs)
        return max(lb, min(ub, rew))
