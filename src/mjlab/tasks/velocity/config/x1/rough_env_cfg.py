from dataclasses import dataclass, replace

from mjlab.asset_zoo.robots.agibot_x1.x1_constants import (
  X1_ACTION_SCALE,
  X1_ROBOT_CFG,
)
from mjlab.tasks.velocity.velocity_env_cfg import (
  LocomotionVelocityEnvCfg,
)
from mjlab.utils.spec_config import ContactSensorCfg



from mjlab.tasks.velocity import mdp
from mjlab.managers.manager_term_config import RewardTermCfg as RewardTerm
from mjlab.managers.manager_term_config import term

@dataclass
class AgibotX1RoughEnvCfg(LocomotionVelocityEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    foot_contact_sensors = [
      ContactSensorCfg(
        name=f"{side}_foot_ground_contact",
        body1=f"{side}_ankle_roll_link",
        body2="terrain",
        num=1,
        data=("found",),
        reduce="netforce",
      )
      for side in ["left", "right"]
    ]
    g1_cfg = replace(X1_ROBOT_CFG, sensors=tuple(foot_contact_sensors))
    self.scene.entities = {"robot": g1_cfg}

    sensor_names = ["left_foot_ground_contact", "right_foot_ground_contact"]
    geom_names = []
    for i in range(1, 8):
      geom_names.append(f"left_foot{i}_collision")
    for i in range(1, 8):
      geom_names.append(f"right_foot{i}_collision")

    self.events.foot_friction.params["asset_cfg"].geom_names = geom_names

    self.actions.joint_pos.scale = X1_ACTION_SCALE

    self.rewards.air_time.params["sensor_names"] = sensor_names

    self.rewards.pose.params["std"] = {
      # Lower body.
      r".*hip_pitch.*": 0.3,
      r".*hip_roll.*": 0.15,
      r".*hip_yaw.*": 0.15,
      r".*knee.*": 0.35,
      r".*ankle_pitch.*": 0.25,
      r".*ankle_roll.*": 0.1,
      # Waist.
      r".*waist_yaw.*": 0.15,
      r".*waist_roll.*": 0.08,
      r".*waist_pitch.*": 0.1,
      # Arms.
      r".*shoulder_pitch.*": 0.35,
      r".*shoulder_roll.*": 0.15,
      r".*shoulder_yaw.*": 0.1,
      r".*elbow.*": 0.25,
    }

    self.viewer.body_name = "torso_link"
    self.commands.twist.viz.z_offset = 1.0

    self.curriculum.command_vel = None
    
    self._add_x1_rewards()
  
  def _add_x1_rewards(self):
    """添加 AgibotX1 特有的奖励函数。"""
    x1_rewards = {
      "foot_clearance": {
        "func": mdp.feet_air_time,
        "weight": 0.25,
        "params": {
          "asset_name": "robot",
          "sensor_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
          "threshold_min": 0.1,
          "threshold_max": 0.5,
          "command_name": "twist",
          "command_threshold": 0.1,
        }
      },
      "feet_slide": {
        "func": mdp.feet_slide,
        "weight": -0.5,
        "params": {
          "asset_name": "robot",
          "sensor_names": ["left_foot_ground_contact", "right_foot_ground_contact"],
        }
      },
    }
    
    for name, config in x1_rewards.items():
      setattr(
        self.rewards,
        name,
        term(RewardTerm, **config)
      )

@dataclass
class AgibotX1RoughEnvCfg_PLAY(AgibotX1RoughEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)

    if self.scene.terrain is not None:
      if self.scene.terrain.terrain_generator is not None:
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.border_width = 10.0
