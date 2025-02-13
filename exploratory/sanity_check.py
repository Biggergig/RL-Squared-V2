
import numpy as np
from rlgym_sim.utils.gamestates import GameState
from rlgym_ppo.util import MetricsLogger


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

from rlgym_sim.utils import RewardFunction, math
from rlgym_sim.utils.gamestates import GameState, PlayerData
class DontMoveReward(RewardFunction):
	def __init__(self):
		super().__init__()

	def reset(self, initial_state: GameState):
		pass

	def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
		z = player.car_data.linear_velocity
		return min(-np.log(np.linalg.norm(z)),10)


def build_rocketsim_env():
	import rlgym_sim
	from rlgym_sim.utils.reward_functions import CombinedReward
	from rlgym_sim.utils.reward_functions.common_rewards import LiuDistancePlayerToBallReward, VelocityPlayerToBallReward, VelocityBallToGoalReward, \
		EventReward
	from rlgym_sim.utils.obs_builders import DefaultObs
	from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition, TimeoutCondition
	from rlgym_sim.utils import common_values
	from rlgym_sim.utils.action_parsers import ContinuousAction

	spawn_opponents = True
	team_size = 1
	game_tick_rate = 120
	tick_skip = 8
	timeout_seconds = 30
	timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

	action_parser = ContinuousAction()
	# terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]
	terminal_conditions = [TimeoutCondition(timeout_seconds)]

	# rewards_to_combine = ()
	# reward_weights = (-1,)

	# reward_fn = CombinedReward(reward_functions=rewards_to_combine,
	# 						   reward_weights=reward_weights)
	reward_fn = DontMoveReward()

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


	# educated guess - could be slightly higher or lower
	# min_inference_size = max(1, int(round(n_proc * 0.9)))

	N_PROC = 2
	TS_PER_ITER = 50_000
	NETWORK_SHAPE = (1024,1024,1024)
	PPO_EPOCHS = 2
	PPO_MINIBATCH_SIZE = 50_000
	PPO_ENT_COEF = .01
	NET_LR = 2e-4
	learner = Learner(build_rocketsim_env,
					  n_proc=N_PROC,
					  render=False,
					  render_delay=0,
					  timestep_limit=1_000_000_000,
					  exp_buffer_size=TS_PER_ITER*3,
					  ts_per_iteration=TS_PER_ITER,
					  policy_layer_sizes=NETWORK_SHAPE,
					  critic_layer_sizes=NETWORK_SHAPE,
					  ppo_epochs=PPO_EPOCHS,
					  ppo_batch_size=TS_PER_ITER,
					  ppo_minibatch_size=PPO_MINIBATCH_SIZE,
					  ppo_ent_coef=PPO_ENT_COEF,
					  critic_lr=NET_LR,
					  policy_lr=NET_LR,
					  log_to_wandb=True,
					  checkpoints_save_folder=None,
					  save_every_ts=150_000,

					  metrics_logger=metrics_logger,
					#   min_inference_size=min_inference_size,
					#   standardize_returns=True,
					#   standardize_obs=False,
					#   save_every_ts=100_000_000,
					  checkpoint_load_folder="latest",
					#   load_wandb=False,
	)
	learner.learn()
