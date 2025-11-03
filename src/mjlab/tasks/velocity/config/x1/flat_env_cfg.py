from dataclasses import dataclass

from mjlab.tasks.velocity.config.x1.rough_env_cfg import (
  AgibotX1RoughEnvCfg,
)


@dataclass
class AgibotX1FlatEnvCfg(AgibotX1RoughEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    assert self.scene.terrain is not None
    self.scene.terrain.terrain_type = "plane"
    self.scene.terrain.terrain_generator = None
    self.curriculum.terrain_levels = None


@dataclass
class AgibotX1FlatEnvCfg_PLAY(AgibotX1FlatEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)

    self.observations.policy.enable_corruption = False
    self.events.push_robot = None

    self.commands.twist.ranges.lin_vel_x = (-1.5, 2.0)
    self.commands.twist.ranges.ang_vel_z = (-0.7, 0.7)
