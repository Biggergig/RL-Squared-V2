import numpy as np
import os
from time import time

from metricLogger import MyMetricLogger
from rewards import *
from build_environment import build_rocketsim_env


def run(config):
    from rlgym_ppo import Learner

    metrics_logger = MyMetricLogger()

    # educated guess - could be slightly higher or lower
    # min_inference_size = max(1, int(round(n_proc * 0.9)))

    CHECKPOINT_LOAD_PATH = config["checkpoint_load_path"]
    if CHECKPOINT_LOAD_PATH is not None:
        CHECKPOINT_LOAD_PATH = os.path.join(
            CHECKPOINT_LOAD_PATH,
            max(os.listdir(CHECKPOINT_LOAD_PATH), key=lambda x: int(x.split("-")[-1])),
        )
        print("Loading from latest checkpoint", CHECKPOINT_LOAD_PATH)

    PHASE = config["phase"]
    TS_PER_ITER = config["ts_per_iter"]
    NETWORK_SHAPE = config["network_shape"]

    learner = Learner(
        build_rocketsim_env,
        n_proc=config["num_processes"],
        render=config["render"],
        render_delay=config["render_delay"],
        timestep_limit=config["timestep_limit"],
        exp_buffer_size=TS_PER_ITER * config["exp_buffer_size_multiple"],
        ts_per_iteration=TS_PER_ITER,
        policy_layer_sizes=NETWORK_SHAPE,
        critic_layer_sizes=NETWORK_SHAPE,
        ppo_epochs=config["ppo_epochs"],
        ppo_batch_size=TS_PER_ITER,
        ppo_minibatch_size=config["ppo_minibatch_size"],
        ppo_ent_coef=config["ppo_ent_coef"],
        policy_lr=config["policy_net_lr"],
        critic_lr=config["critic_net_lr"],
        log_to_wandb=config["log_to_wandb"],
        wandb_run_name=f"{config["wandb_run_name_prefix"]}{PHASE}{int(time()) % 1_000_000}",
        wandb_project_name=f"{config["wandb_project_name_prefix"]}{PHASE}",
        checkpoints_save_folder=config["checkpoint_save_folder_prefix"]
        + f"{PHASE}/ppo",
        save_every_ts=config["save_every_ts"],
        n_checkpoints_to_keep=100_000,
        metrics_logger=metrics_logger,
        checkpoint_load_folder=CHECKPOINT_LOAD_PATH,
        load_wandb=config["load_wandb"],
    )

    learner.learn()


if __name__ == "__main__":
    from sys import argv
    import yaml

    configName = argv[1] if len(argv) > 1 else "config.yaml"
    config = None
    with open(configName) as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    if config is None:
        print("Failed to load config", configName)
        exit(1)

    print("Loading config into env variables...")
    for key, val in config.items():
        os.environ[f"RLBOT_{key}"] = str(val)

    run(config)
