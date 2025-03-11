import os
import arguably
import yaml
from rlgym_ppo import Learner
from utils import *
import time
import contextlib


@arguably.command
def run(config: str = "default.yaml"):
    with open(config, "r") as f:
        config_dict = yaml.safe_load(f)
    config_dict["exp_buffer_size"] = (
        int(config_dict["_exp_buffer_size_multiple"]) * config_dict["ts_per_iteration"]
    )

    config_dict["ppo_batch_size"] = config_dict["ts_per_iteration"]
    config_dict["_RUN_ID"] = str(int(time.time()))

    config_dict["checkpoints_save_folder"] = os.path.join(
        "data",
        "checkpoints",
        config_dict['_name'],
        str(config_dict['_phase']),
    )

    inp_cfg = {k:v for k,v in config_dict.items() if not k.startswith("_")}
    # print(config_dict)
    # print(inp_cfg)

    os.environ["RLBOT_NAME"] = config_dict["_name"]
    os.environ["RLBOT_PHASE"] = str(config_dict["_phase"])
    os.environ["RLBOT_LOG_TO_WANDB"] = str(config_dict["log_to_wandb"])
    os.environ["RLBOT_RUN_ID"] = str(config_dict["_RUN_ID"])

    inp_cfg["wandb_run"] = load_run(config=config_dict, reward_fn=False)

    with contextlib.suppress(FileNotFoundError): # lock for only one logger
        os.remove(".rew_set_global.tmp")

    learner = Learner(
        build_rocketsim_env,
        **inp_cfg
    )

    learner.learn()


if __name__ == "__main__":
    arguably.run()
