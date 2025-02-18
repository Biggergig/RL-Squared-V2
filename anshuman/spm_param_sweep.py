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
    print("Run Name:", config["wandb_run_name_prefix"])

    sweep_config = {
        "ppo_minibatch_size": [50_000],
        "ts_per_iter": [100_000],
        "timestep_limit": [150_000],
        "num_processes": [40, 48],
        "network_shape": [(2048, 2048, 1024, 1024)],
    }

    keys = list(sweep_config.keys())

    def combos(cfg, pos=0):
        if pos == len(keys):
            yield dict(cfg)
            return
        key = keys[pos]
        for v in sweep_config[key]:
            cfg[key] = v
            yield from combos(cfg, pos + 1)

    for cfg in combos(config):
        run(cfg)
