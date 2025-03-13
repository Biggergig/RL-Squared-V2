import time
import numpy as np

import rlgym_sim

import rlgym_sim
from rlgym_sim.utils.obs_builders import DefaultObs
from rlgym_sim.utils.terminal_conditions.common_conditions import *
from rlgym_sim.utils import common_values
from rlgym_sim.utils.action_parsers import DiscreteAction
from rlgym_sim.utils.reward_functions.common_rewards import EventReward

game_tick_rate = 120
tick_skip = 8
TPS = game_tick_rate / tick_skip


def build_env():
    timeout_seconds = 5
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    action_parser = DiscreteAction()
    terminal_conditions = [
        # NoTouchTimeoutCondition(timeout_ticks),
        GoalScoredCondition(),
    ]

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
        team_size=1,
        spawn_opponents=True,
        terminal_conditions=terminal_conditions,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=EventReward(goal=1),
    )

    return env


env = build_env()
goals = [0, 0, 0]
for _ in range(10):
    obs = env.reset()

    done = False
    steps = 0
    t0 = time.time()
    starttime = time.time()
    while not done:
        actions_1 = env.action_space.sample()
        actions_2 = env.action_space.sample()
        actions = np.vstack([actions_1, actions_2])
        new_obs, reward, done, state = env.step(actions)
        env.render()
        for i in range(2):
            goals[i] += reward[i]
        steps += 1
        # print(env.step(actions))
        # Sleep to keep the game in real time
        # time.sleep(max(0, starttime + steps / TPS - time.time()))
    goals[int(state["result"]) + 1] += 1
    print(goals)

    length = time.time() - t0
    print(
        "Step time: {:1.5f} | Episode time: {:.2f} | Episode Reward: {}".format(
            length / steps, length, goals
        )
    )
