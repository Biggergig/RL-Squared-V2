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
    timeout_seconds = 120
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    action_parser = DiscreteAction()
    terminal_conditions = [
        GoalScoredCondition(),
    ]

    rewards = (
        (
            EventReward(
                goal=10, concede=-10, shot=3, save=3, boost_pickup=0.1, touch=0.25
            ),
            1,
        ),
        (VelocityBallToGoalReward(), 5),
        (VelocityBallToGoalReward(own_goal=True), -3),
        (LiuDistanceBallToGoalReward(), 5),
        (LiuDistanceBallToGoalReward(own_goal=True), -3),
        (VelocityPlayerToBallReward(), 2),
        (FaceBallReward(), 0.5),
        (InAirReward(), 0.15),
    )

    reward_fn = CombinedReward.from_zipped(*rewards)

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


if __name__ == "__main__":
    from rlgym_ppo import Learner
    from sys import argv

    argv = [s.lower() for s in argv]

    DEBUG = "debug" in argv or "dry" in argv
    if DEBUG:
        print("IN DEBUG MODE")

    NEWRUN = "new" in argv
    if NEWRUN:
        print("Creating new run")

    NEWWANDB = "newwandb" in argv or NEWRUN
    if NEWWANDB:
        print("Creating new wandb run")

    metrics_logger = MyMetricLogger()

    N_PROC = 48
    min_inference_size = max(1, int(round(N_PROC * 0.9)))
    TS_PER_ITER = 100_000
    NETWORK_SHAPE = (2048, 2048, 1024, 1024)
    PPO_EPOCHS = 2
    PPO_MINIBATCH_SIZE = 50_000  # test for SPM
    PPO_ENT_COEF = 0.01
    NET_LR = 2e-4
    SAVE_EVERY_TS = 250_000
    CHECKPOINT_SAVE_FOLDER = "data/checkpoints/p2/ppo"
    CHECKPOINT_LOAD_DIR = None
    for loadFrom in (2, 1, 0):
        try:
            CHECKPOINT_LOAD_DIR = f"data/checkpoints/p{loadFrom}/" + str(
                max(
                    os.listdir(f"data/checkpoints/p{loadFrom}/"),
                    key=lambda d: int(d.split("-")[-1]),
                )
            )
            CHECKPOINT_LOAD_DIR += "/" + max(
                os.listdir(CHECKPOINT_LOAD_DIR), key=lambda d: int(d.split("-")[-1])
            )
            if loadFrom != 2:
                NEWWANDB = True
                print("Setting new wandb since loading from prior phase")
            break
        except (ValueError, FileNotFoundError):
            print("No checkpoints from phase", loadFrom)

    if DEBUG:
        N_PROC = 1
        SAVE_EVERY_TS = 1_000_000_000_000
        CHECKPOINT_SAVE_FOLDER = "data/debug/checkpoints"
        CHECKPOINT_LOAD_DIR = None

    learner = Learner(
        build_rocketsim_env,
        n_proc=N_PROC,
        render=True,
        render_delay=0,
        timestep_limit=100_000_000_000_000,
        exp_buffer_size=TS_PER_ITER * 3,
        ts_per_iteration=TS_PER_ITER,
        policy_layer_sizes=NETWORK_SHAPE,
        critic_layer_sizes=NETWORK_SHAPE,
        ppo_epochs=PPO_EPOCHS,
        ppo_batch_size=TS_PER_ITER,
        ppo_minibatch_size=PPO_MINIBATCH_SIZE,
        ppo_ent_coef=PPO_ENT_COEF,
        critic_lr=NET_LR,
        policy_lr=0,
        log_to_wandb=not DEBUG,
        wandb_run_name="PPO-p2-" + str(int(time()) % 1_000_000),
        wandb_project_name="ans_phase2",
        checkpoints_save_folder=CHECKPOINT_SAVE_FOLDER,
        save_every_ts=SAVE_EVERY_TS,
        n_checkpoints_to_keep=100_000,
        metrics_logger=metrics_logger,
        checkpoint_load_folder=CHECKPOINT_LOAD_DIR,
        load_wandb=not (DEBUG or NEWWANDB),
        min_inference_size=min_inference_size,
        standardize_returns=True,
        #   standardize_obs=False,
    )

    with open(__file__, "r") as f:
        print(" --- CURRENT FILE --- ")
        print(f.read())

    if "dry" not in argv:
        learner.learn()

# Goal is to standardize returns and retrain just critic
