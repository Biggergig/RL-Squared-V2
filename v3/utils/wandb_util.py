import os
import wandb

from wandb.sdk.wandb_run import Run


def load_run(config=None, reward_fn=False) -> Run | None:
    NAME = os.environ["RLBOT_NAME"]
    PHASE = int(os.environ["RLBOT_PHASE"])

    run_name = "Rewards" if reward_fn else "PPO"
    project_name = f"{NAME}_phase{PHASE}"
    group_name = os.environ["RLBOT_RUN_ID"]

    if os.environ["RLBOT_LOG_TO_WANDB"] == "True":
        wandb_run = wandb.init(
            entity="rl-squared",
            project=project_name,
            group=group_name,
            config=config,
            name=run_name,
            reinit=False,
        )
        return wandb_run
    return None
