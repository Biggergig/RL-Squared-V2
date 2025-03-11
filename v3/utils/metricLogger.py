from rlgym_sim.utils.gamestates import GameState
from rlgym_ppo.util import MetricsLogger
from .rewards import DistancePlayerToBallReward


class MyMetricLogger(MetricsLogger):
    def __init__(self):
        self.distToBallReward = DistancePlayerToBallReward()

    def _collect_metrics(self, game_state: GameState) -> list:
        return [
            game_state.orange_score + game_state.blue_score,
            sum(
                self.distToBallReward.get_reward(player, game_state, None)
                for player in game_state.players
            )
            / len(game_state.players),
            sum(player.ball_touched for player in game_state.players)
            / len(game_state.players),
        ]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avgs = [0, 0, 0]
        for metric_array in collected_metrics:
            for i in range(len(avgs)):
                avgs[i] += metric_array[i]
        for i in range(len(avgs)):
            avgs[i] /= len(collected_metrics)

        report = {
            "1. Average Total Score": avgs[0],
            "2. Average total distance to ball": avgs[1],
            "3. Average times ball touched": avgs[2],
            "4. Cumulative Timesteps": cumulative_timesteps,
            "5. Num Metrics": len(collected_metrics),
        }
        wandb_run.log(report)
