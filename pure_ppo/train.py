import os
import numpy as np
from rlgym_sim.utils.gamestates import GameState
from rlgym_ppo.util import MetricsLogger

import rlgym_sim
from rlgym_sim.utils.reward_functions import CombinedReward
from rlgym_sim.utils.reward_functions.common_rewards import *
from rlgym_sim.utils.obs_builders import DefaultObs
from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
from rlgym_sim.utils import common_values
from rlgym_sim.utils.action_parsers import ContinuousAction
from rlgym_sim.utils.action_parsers import DiscreteAction
# from rewards.CustomRewards import *


class ExampleLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        return [game_state.players[0].car_data.linear_velocity,
                game_state.players[0].car_data.rotation_mtx(),
                game_state.orange_score]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avg_linvel = np.zeros(3)
        for metric_array in collected_metrics:
            p0_linear_velocity = metric_array[0]
            avg_linvel += p0_linear_velocity
        avg_linvel /= len(collected_metrics)
        report = {"x_vel":avg_linvel[0],
                  "y_vel":avg_linvel[1],
                  "z_vel":avg_linvel[2],
                  "Cumulative Timesteps":cumulative_timesteps}
        wandb_run.log(report)


def build_rocketsim_env():
    spawn_opponents = True
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 30
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    action_parser = DiscreteAction()
    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]

    stage = 0

    if stage == 0:
        rewards = (
            (EventReward(team_goal=1, concede=-1), 10),
            (ConstantReward(), -1),
        )

    reward_fn = CombinedReward.from_zipped(*rewards)

    obs_builder = DefaultObs(
        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL)

    env = rlgym_sim.make(tick_skip=tick_skip,
                         team_size=team_size,
                         spawn_opponents=spawn_opponents,
                         terminal_conditions=terminal_conditions,
                         reward_fn=reward_fn,
                         obs_builder=obs_builder,
                         action_parser=action_parser)

    return env

if __name__ == "__main__":
    from rlgym_ppo import Learner
    metrics_logger = ExampleLogger()

    # num processes processes
    n_proc = 32

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    # latest checkpoint
    latest_checkpoint_dir = "data\checkpoints"
    latest_checkpoint_dir = os.path.join(latest_checkpoint_dir, max(os.listdir(latest_checkpoint_dir), key=lambda d: int(d.split("-")[-1])))
    print(latest_checkpoint_dir)
    latest_checkpoint_dir = os.path.join(latest_checkpoint_dir, max(os.listdir(latest_checkpoint_dir), key=lambda d: int(d.split("-")[-1])))

    learner = Learner(build_rocketsim_env,
                      n_proc=n_proc,
                      render=True,
                      render_delay=0,
                      min_inference_size=min_inference_size,
                      metrics_logger=metrics_logger,
                      ppo_batch_size=100000,
                      ts_per_iteration=100000,
                      exp_buffer_size=300000,
                      ppo_minibatch_size=50000,
                      ppo_ent_coef=0.01,
                      ppo_epochs=3,
                      policy_lr=2e-4,
                      critic_lr=2e-4,
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=1_000_000,
                      timestep_limit=10e10,
                      policy_layer_sizes=(1024,1024,512,512),
                      critic_layer_sizes=(1024,1024,512,512),
                      checkpoint_load_folder=latest_checkpoint_dir,
                      log_to_wandb=True)
    learner.learn()