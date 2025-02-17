from rlgym_sim.utils.gamestates import GameState
from rlgym_ppo.util import MetricsLogger
from rewards import DistancePlayerToBallReward


class MyMetricLogger(MetricsLogger):
    def __init__(self):
        self.distToBallReward = DistancePlayerToBallReward()

    def _collect_metrics(self, game_state: GameState) -> list:
        return [
            game_state.orange_score + game_state.blue_score,
            1.0 if game_state.players[0].ball_touched else 0.0,
            1.0 if game_state.players[1].ball_touched else 0.0,
            sum(
                self.distToBallReward.get_reward(player, game_state, None)
                for player in game_state.players
            )
            / len(game_state.players),
        ]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avgs = [0, 0, 0, 0]
        for metric_array in collected_metrics:
            for i in range(len(avgs)):
                avgs[i] += metric_array[i]
        for i in range(len(avgs)):
            avgs[i] /= len(collected_metrics)
        report = {
            "1. Average Total Score": avgs[0],
            "2. Player 0 ball touch averages": avgs[1],
            "3. Player 1 ball touch averages": avgs[2],
            "4. Average total distance to ball": avgs[3],
            "5. Cumulative Timesteps": cumulative_timesteps,
            "6. Num Metrics": len(collected_metrics),
        }
        wandb_run.log(report)
