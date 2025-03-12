from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState


class FnReward(RewardFunction):
    def __init__(
        self,
        reward_fn: RewardFunction,
        scale_fn,
        n_proc: int = 1,
    ):
        super().__init__()
        self.fn = reward_fn
        self.scale = scale_fn
        self.ts = 0
        self.n_proc = n_proc

    # Called when the game resets (i.e. after a goal is scored)
    def reset(self, initial_state: GameState):
        self.fn.reset(initial_state)

    # Get the reward for a specific player, at the current state
    def get_reward(self, *args, **kwargs) -> float:
        rew = self.fn.get_reward(*args, **kwargs)
        self.ts += 1
        return self.scale(rew, self.ts * self.n_proc)

    def get_final_reward(self, *args, **kwargs) -> float:
        rew = self.fn.get_final_reward(*args, **kwargs)
        self.ts += 1
        return self.scale(rew, self.ts * self.n_proc)
