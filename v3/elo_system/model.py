import torch
import json
import os
from .elo_env import build_env
from rlgym_ppo.ppo.multi_discrete_policy import MultiDiscreteFF

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
default_env = build_env()


class Model:
    def __init__(self, path, name, deterministic=False):
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
        layer_size = BOOK_KEEPING["wandb_config"]["policy_layer_sizes"]
        self.model = MultiDiscreteFF(inp_shape, layer_size, device)
        self.model.load_state_dict(torch.load(os.path.join(path, "PPO_POLICY.pt")))

    def act(self, obs):
        if self.model is None:
            return self.outspace.sample() * 0
        return self.model.get_action(obs, deterministic=self.deterministic)[0]
