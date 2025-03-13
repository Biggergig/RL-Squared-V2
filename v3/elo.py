import time
import numpy as np
import json
import os
import torch

import rlgym_sim

import rlgym_sim
from rlgym_sim.utils.obs_builders import DefaultObs
from rlgym_sim.utils.terminal_conditions.common_conditions import *
from rlgym_sim.utils import common_values
from rlgym_sim.utils.action_parsers import DiscreteAction
from rlgym_sim.utils.reward_functions.common_rewards import EventReward
from rlgym_ppo.ppo.multi_discrete_policy import MultiDiscreteFF

game_tick_rate = 120
tick_skip = 8
TPS = game_tick_rate / tick_skip


def build_env():
    timeout_seconds = 60
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    action_parser = DiscreteAction()
    terminal_conditions = [
        NoTouchTimeoutCondition(timeout_ticks),
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


model_1_path = "data/checkpoints/curriculum/remote/curriculum/1/405360906"
model_2_path = "data/checkpoints/curriculum/remote/curriculum/1/102016886"


class Model:
    def __init__(self, path, inp_shape, action_space, device):
        if path is None:
            self.model = None
            self.outspace = action_space
            return

        BOOK_KEEPING = json.load(
            open(os.path.join(path, "BOOK_KEEPING_VARS.json"), "r")
        )
        layer_size = BOOK_KEEPING["wandb_config"]["policy_layer_sizes"]
        self.model = MultiDiscreteFF(inp_shape, layer_size, device)
        self.model.load_state_dict(torch.load(os.path.join(path, "PPO_POLICY.pt")))

    def act(self, obs):
        if self.model is None:
            return self.outspace.sample() * 0
        return self.model.get_action(obs, deterministic=True)[0]


env = build_env()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
models = [
    Model(p, env.observation_space.shape[0], env.action_space, device)
    for p in (model_1_path, model_2_path)
]
# print([m.model for m in models])
# exit()
goals = [0, 0, 0]
# orange, timeout, blue
for _ in range(10):
    obs = env.reset()

    done = False
    steps = 0
    t0 = time.time()
    starttime = time.time()
    while not done:
        actions = np.vstack([m.act(obs[i]) for i, m in enumerate(models)])
        obs, reward, done, state = env.step(actions)
        env.render()
        steps += 1
        # print(env.step(actions))
        # Sleep to keep the game in real time
        # time.sleep(max(0, starttime + steps / TPS - time.time()))
    goals[-int(state["result"]) + 1] += 1
    length = time.time() - t0
    print(
        "Step time: {:1.5f} | Episode time: {:.2f} | Goals: {}".format(
            length / steps, length, goals
        )
    )
env.close()
