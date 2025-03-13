import torch
import json
import os
from .elo_env import build_env
from .consts import USE_GPU
from rlgym_ppo.ppo.multi_discrete_policy import MultiDiscreteFF

device = (
    torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if USE_GPU else "cpu"
)

default_env = build_env()


class Model:
    def __init__(self, path, name=None, deterministic=False):
        # 400m curriculum not deterministic beat deterministic
        self.name = name
        inp_shape = default_env.observation_space.shape[0]
        action_space = default_env.action_space
        if path is None:
            self.model = None
            self.outspace = action_space
            return

        self.deterministic = deterministic
        BOOK_KEEPING = json.load(
            open(os.path.join(path, "BOOK_KEEPING_VARS.json"), "r")
        )

        WANDB_CONFIG = BOOK_KEEPING["wandb_config"]
        policy_shape = WANDB_CONFIG.get("policy_layer_sizes", None)
        if policy_shape is None:
            policy_shape = WANDB_CONFIG.get("policy_network_shape", None)
        if policy_shape is None:
            policy_shape = WANDB_CONFIG.get("network_shape", None)
        assert policy_shape is not None, "No policy shape found"

        self.model = MultiDiscreteFF(inp_shape, policy_shape, device)
        self.model.load_state_dict(torch.load(os.path.join(path, "PPO_POLICY.pt")))
        # print(self.name, "is loaded onto", device)
        if self.name is None:
            self.steps = BOOK_KEEPING["cumulative_timesteps"]
            self.shape = policy_shape
            self.param_count = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            self.name = f"{'_'.join(BOOK_KEEPING['wandb_project'].split('_')[1:-1])}_{BOOK_KEEPING["cumulative_timesteps"]//1_000_000}m"

    def act(self, obs):
        if self.model is None:
            return self.outspace.sample() * 0
        return self.model.get_action(obs, deterministic=self.deterministic)[0]
