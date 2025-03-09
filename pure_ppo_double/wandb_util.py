import os
import wandb
import time


def load_run(reinit=False, config=None):
    PHASE = os.environ["RLBOT_PHASE"]

    wandb_run_name = f"{os.environ['RLBOT_WANDB_RUN_NAME_PREFIX']}{PHASE}{os.environ['RLBOT_RUN_ID']}"
    wandb_project_name = f"{os.environ['RLBOT_WANDB_PROJECT_NAME_PREFIX']}{PHASE}"
    wandb_group_name = os.environ["RLBOT_RUN_ID"]

    if os.environ["RLBOT_LOG_TO_WANDB"] == "True":
        project = "rlgym-ppo" if wandb_project_name is None else wandb_project_name
        group = "unnamed-runs" if wandb_group_name is None else wandb_group_name
        run_name = "rlgym-ppo-run" if wandb_run_name is None else wandb_run_name
        wandb_run = wandb.init(
            entity="rl-squared",
            project=project,
            group=group,
            config=config,
            name=run_name,
            reinit=reinit,
        )
        return wandb_run
    return None
