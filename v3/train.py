import os
import arguably
import yaml
from rlgym_ppo import Learner
from utils import *


@arguably.command
def run(config: str = "default.yaml"):
    with open(config, "r") as f:
        config_dict = yaml.safe_load(f)
    config_dict["exp_buffer_size"] = (
        int(config_dict["_exp_buffer_size_multiple"]) * config_dict["ts_per_iteration"]
    )
    print(config_dict)

    inp_cfg = {k:v for k,v in config_dict.items() if not k.startswith("_")}
    print(inp_cfg)

    learner = Learner(
        build_rocketsim_env,
        **inp_cfg
    )


if __name__ == "__main__":
    arguably.run()
