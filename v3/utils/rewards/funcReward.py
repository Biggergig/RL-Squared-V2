import os
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState


class FnReward(RewardFunction):
    def __init__(
        self,
        reward_fn: RewardFunction,
        scale_fn,
    ):
        super().__init__()
        self.fn = reward_fn
        self.scale = scale_fn
        self.ts = 0
        self.n_proc = int(os.environ["RLBOT_N_PROC"])

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


class TimeReward(RewardFunction):
    def __init__(
        self,
        reward_fn: RewardFunction,
        time_fn,
    ):
        super().__init__()
        self.fn = reward_fn
        self.time_fn = time_fn
        self.ts = 0
        self.n_proc = int(os.environ["RLBOT_N_PROC"])

    # Called when the game resets (i.e. after a goal is scored)
    def reset(self, initial_state: GameState):
        self.fn.reset(initial_state)

    # Get the reward for a specific player, at the current state
    def get_reward(self, *args, **kwargs) -> float:
        rew = self.fn.get_reward(*args, **kwargs)
        self.ts += 1
        return rew * self.time_fn(self.ts * self.n_proc)

    def get_final_reward(self, *args, **kwargs) -> float:
        rew = self.fn.get_final_reward(*args, **kwargs)
        self.ts += 1
        return rew * self.time_fn(self.ts * self.n_proc)


def curriculum(start, m1, m2, end2):
    def fn(t):
        if t < start or t > end2:
            return 0
        if m1 <= t <= m2:
            return 1
        if start <= t < m1:
            return (t - start) / (m1 - start)
        if m2 <= t < end2:
            return (end2 - t) / (end2 - m2)
        return -1

    return fn
