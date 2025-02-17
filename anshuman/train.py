import numpy as np

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
    timeout_seconds = 60
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    action_parser = DiscreteAction()
    terminal_conditions = [
        NoTouchTimeoutCondition(timeout_ticks),
        GoalScoredCondition(),
    ]

    rewards = (
        (EventReward(touch=1), 50),
        (VelocityPlayerToBallReward(), 5),
        (FaceBallReward(), 1),
        (InAirReward(), 0.15),
    )

    reward_fn = CombinedReward(
        reward_functions=[r[0] for r in rewards], reward_weights=[r[1] for r in rewards]
    )

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
            True, True, True
        ),  # Randomized speeds for cars and ball, but on floor
    )

    return env


if __name__ == "__main__":
    from rlgym_ppo import Learner
    from sys import argv

    DEBUG = "debug" in argv or "dry" in argv
    if DEBUG:
        print("IN DEBUG MODE")

    metrics_logger = MyMetricLogger()

    # educated guess - could be slightly higher or lower
    # min_inference_size = max(1, int(round(n_proc * 0.9)))

    N_PROC = 48
    TS_PER_ITER = 50_000
    NETWORK_SHAPE = (2048, 2048, 1024, 1024)
    PPO_EPOCHS = 2
    PPO_MINIBATCH_SIZE = 50_000
    PPO_ENT_COEF = 0.01
    NET_LR = 2e-4

    if DEBUG:
        N_PROC = 1

    learner = Learner(
        build_rocketsim_env,
        n_proc=N_PROC,
        render=True,
        render_delay=0,
        timestep_limit=1_000_000_000,
        exp_buffer_size=TS_PER_ITER * 3,
        ts_per_iteration=TS_PER_ITER,
        policy_layer_sizes=NETWORK_SHAPE,
        critic_layer_sizes=NETWORK_SHAPE,
        ppo_epochs=PPO_EPOCHS,
        ppo_batch_size=TS_PER_ITER,
        ppo_minibatch_size=PPO_MINIBATCH_SIZE,
        ppo_ent_coef=PPO_ENT_COEF,
        critic_lr=NET_LR,
        policy_lr=NET_LR,
        log_to_wandb=not DEBUG,
        checkpoints_save_folder=None,
        save_every_ts=150_000 if not DEBUG else 1_000_000_000_000,
        n_checkpoints_to_keep=100_000,
        metrics_logger=metrics_logger,
        #   min_inference_size=min_inference_size,
        #   standardize_returns=True,
        #   standardize_obs=False,
        checkpoint_load_folder="latest" if not DEBUG else None,
        load_wandb=False,
    )

    with open(__file__, "r") as f:
        print(" --- CURRENT FILE --- ")
        print(f.read())

    if "dry" not in argv:
        learner.learn()
