import numpy as np
import os
from time import time

from metricLogger import MyMetricLogger
from rewards import *

import rlgym_sim
from rlgym_sim.utils.reward_functions import CombinedReward
from rlgym_sim.utils.reward_functions.common_rewards import *
from rlgym_sim.utils.obs_builders import DefaultObs
from rlgym_sim.utils.terminal_conditions.common_conditions import *
from rlgym_sim.utils import common_values
from rlgym_sim.utils.action_parsers import DiscreteAction
from rlgym_sim.utils.state_setters import RandomState


def build_rocketsim_env():
    spawn_opponents = True
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 30
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    phase = os.environ["RLBOT_PHASE"]
    try:
        phase = int(phase)
    except ValueError:
        print("Phase not set")
        phase = -1
    print("Environment phase:", phase)

    action_parser = DiscreteAction()
    terminal_conditions = [
        NoTouchTimeoutCondition(timeout_ticks),
        GoalScoredCondition(),
    ]

    if phase == 0:
        rewards = (
            (EventReward(touch=0.5), 1),
            (EventReward(shot=5), 1),
            (EventReward(goal=20), 1),
            (EventReward(concede=-20), 1),
            (VelocityPlayerToBallReward(), 3),
            (VelocityBallToGoalReward(), 3),
            (FaceBallReward(), 0.1),
            (AlignBallGoal(), 0.3),
            (ConstantReward(), -3),
        )
    elif phase == 1:
        rewards = (
            (EventReward(touch=0.5), 1),
            (EventReward(goal=20), 1),
            (EventReward(concede=-20), 1),
            (VelocityBallToGoalReward(), 5),
            (InAirReward(), 0.15)
            (ConstantReward(), -3),
        )
    else:
        rewards = ((EventReward(boost_pickup=1), 1),)

    reward_fn = CombinedRewardLog.from_zipped(*rewards)

    obs_builder = DefaultObs(
        pos_coef=np.asarray(
            [
                1 / common_values.SIDE_WALL_X,
                1 / common_values.BACK_NET_Y,
                1 / common_values.CEILING_Z,
            ]
        ),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
    )

    env = rlgym_sim.make(
        tick_skip=tick_skip,
        team_size=team_size,
        spawn_opponents=spawn_opponents,
        terminal_conditions=terminal_conditions,
        reward_fn=reward_fn,
        obs_builder=obs_builder,
        action_parser=action_parser,
        state_setter=RandomState(
            True, True, False
        ),  # Randomized speeds for cars + ball, and can be in the air
    )

    return env
