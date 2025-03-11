from .rewards import *
from rlgym_sim.utils.reward_functions import CombinedReward
from rlgym_sim.utils.reward_functions.common_rewards import *


def model_selector(model_name, phase):
    rewards = ((VelocityReward(negative=True), 10, "neg_vel_reward"),)

    if model_name == "gentle":
        if phase == 1:
            # print("GENTLE_PHASE1")
            rewards = (
                (EventReward(touch=1), 1, "touch"),
                (EventReward(team_goal=15), 1, "goal"),
                (EventReward(concede=-10), 1, "enemy_goal"),
                (VelocityPlayerToBallReward(), 1),
                (
                    VelocityBallToGoalReward(use_scalar_projection=True),
                    1 / 5,
                    "scalar_vel_ball_to_goal",
                ),
                (FaceBallReward(), 0.05),
                (AlignBallGoal(), 0.1),
                (SaveBoostReward(), 0.1),
            )

    reward_fn = CombinedRewardLog.from_zipped(*rewards)

    return reward_fn
