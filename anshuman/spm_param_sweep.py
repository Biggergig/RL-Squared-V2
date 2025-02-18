from sys import argv
import yaml
from train import run
from time import time

if __name__ == "__main__":
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

    config["wandb_run_name_prefix"] += str(int(time())) + "-"

    sweep_config = {
        "num_processes": [32, 48, 64],
        "network_shape": [(2048, 2048, 1024, 1024)],
    }

    for key, value_list in sweep_config.items():
        for v in value_list:
            cfg = dict(config)
            cfg[key] = v
            print(cfg)
            run(cfg)
