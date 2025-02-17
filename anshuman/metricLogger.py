from rlgym_sim.utils.gamestates import GameState
from rlgym_ppo.util import MetricsLogger


class MyMetricLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        return [game_state.orange_score + game_state.blue_score]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avg_total_score = 0
        for metric_array in collected_metrics:
            avg_total_score += metric_array[0]
        avg_total_score /= len(collected_metrics)
        report = {
            "Average Total Score": avg_total_score,
            "Cumulative Timesteps": cumulative_timesteps,
        }
        wandb_run.log(report)
