from typing import Any, Optional, Tuple, overload, Union
import os

from pathlib import Path
import contextlib

import numpy as np
from rlgym_sim.utils.reward_functions import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from wandb_util import load_run
from time import sleep
import random


class CombinedRewardLog(RewardFunction):
    """
    A reward composed of multiple rewards.
    """

    def __init__(
        self,
        reward_functions: Tuple[RewardFunction, ...],
        reward_weights: Optional[Tuple[float, ...]] = None,
        log_period=1000,
    ):
        """
        Creates the combined reward using multiple rewards, and a potential set
        of weights for each reward.

        :param reward_functions: Each individual reward function.
        :param reward_weights: The weights for each reward.
        """
        super().__init__()

        self.reward_functions = reward_functions
        self.reward_weights = reward_weights or np.ones_like(reward_functions)

        # self.out = open(str(id(self)) + out, "w")
        # self.out.write(",".join([fn.__name__ for fn in self.reward_functions]))
        if len(self.reward_functions) != len(self.reward_weights):
            raise ValueError(
                (
                    "Reward functions list length ({0}) and reward weights "
                    "length ({1}) must be equal"
                ).format(len(self.reward_functions), len(self.reward_weights))
            )
        sleep(random.random()*5) # just to avoid race
        self.cleaned_up = True
        if os.environ.get("RLBOT_LOG_REWARDS", "False") != "False" and not os.path.exists(".rew_set_global.tmp"):
            Path('.rew_set_global.tmp').touch()
            self.cleaned_up = False
            self.wandb_run = load_run(reinit=False, reward_fn=True)
        else:
            self.wandb_run = None
        self.log_period = log_period
        self.iter = 0

    @classmethod
    def from_zipped(
        cls,
        *rewards_and_weights: Union[RewardFunction, Tuple[RewardFunction, float]],
    ) -> "CombinedReward":
        """
        Alternate constructor which takes any number of either rewards, or (reward, weight) tuples.

        :param rewards_and_weights: a sequence of RewardFunction or (RewardFunction, weight) tuples
        """
        rewards = []
        weights = []
        for value in rewards_and_weights:
            if isinstance(value, tuple):
                r, w = value
            else:
                r, w = value, 1.0
            rewards.append(r)
            weights.append(w)
        return cls(tuple(rewards), tuple(weights))

    def reset(self, initial_state: GameState) -> None:
        """
        Resets underlying reward functions.

        :param initial_state: The initial state of the reset environment.
        """
        for func in self.reward_functions:
            func.reset(initial_state)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        """
        Returns the reward for a player on the terminal state.

        :param player: Player to compute the reward for.
        :param state: The current state of the game.
        :param previous_action: The action taken at the previous environment step.

        :return: The combined rewards for the player on the state.
        """
        rewards = [
            func.get_reward(player, state, previous_action)
            for func in self.reward_functions
        ]

        if self.wandb_run is not None:

            if self.cleaned_up == False:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(".rew_set_global.tmp")
                self.cleaned_up = True
            if self.iter % self.log_period == 0:
                log_dict = {
                    f"rewards/{i+1}_"
                    + (type(self.reward_functions[i]).__name__): rewards[i]
                    * self.reward_weights[i]
                    for i in range(len(self.reward_functions))
                }
                log_dict["rewards/0_TotalReward"] = float(
                    np.dot(self.reward_weights, rewards)
                )
                self.wandb_run.log(log_dict)
                self.iter = 0
            self.iter += 1

        return float(np.dot(self.reward_weights, rewards))

    def get_final_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        """
        Returns the reward for a player on the terminal state.

        :param player: Player to compute the reward for.
        :param state: The current state of the game.
        :param previous_action: The action taken at the previous environment step.

        :return: The combined rewards for the player on the state.
        """
        rewards = [
            func.get_final_reward(player, state, previous_action)
            for func in self.reward_functions
        ]

        if self.wandb_run is not None:
            log_dict = {
                f"rewards/{i+1}_"
                + (type(self.reward_functions[i]).__name__): rewards[i]
                * self.reward_weights[i]
                for i in range(len(self.reward_functions))
            }
            log_dict["rewards/0_TotalReward"] = float(
                np.dot(self.reward_weights, rewards)
            )
            self.wandb_run.log(log_dict)

        return float(np.dot(self.reward_weights, rewards))
