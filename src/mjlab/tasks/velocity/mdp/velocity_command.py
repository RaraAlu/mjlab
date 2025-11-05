from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg
from mjlab.third_party.isaaclab.isaaclab.utils.math import (
  matrix_from_quat,
  quat_apply,
  wrap_to_pi,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class UniformVelocityCommand(CommandTerm):
  """Uniform velocity command generator for robot locomotion.
  
  Generates random velocity commands (linear and angular) for reinforcement
  learning environments with support for heading control and standing modes.
  """
  
  cfg: UniformVelocityCommandCfg

  def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    # Configuration validation.
    if self.cfg.heading_command and self.cfg.ranges.heading is None:
      raise ValueError("heading_command=True but ranges.heading is set to None.")
    if self.cfg.ranges.heading and not self.cfg.heading_command:
      raise ValueError("ranges.heading is set but heading_command=False.")

    # Get robot entity from scene.
    self.robot: Entity = env.scene[cfg.asset_name]

    # Command buffers: [num_envs, 3] where dim 3 = (vx, vy, ωz).
    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    
    # Heading control buffers.
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.heading_error = torch.zeros(self.num_envs, device=self.device)
    
    # Environment mode flags.
    self.is_heading_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_standing_env = torch.zeros_like(self.is_heading_env)

    # Tracking metrics.
    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    """Returns the current velocity command in body frame [num_envs, 3]."""
    return self.vel_command_b

  def _update_metrics(self) -> None:
    """Accumulate velocity tracking errors normalized by command duration."""
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    
    # XY plane linear velocity error.
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], 
        dim=-1
      )
      / max_command_step
    )
    
    # Yaw angular velocity error.
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Resample velocity commands for specified environments.
    
    Args:
        env_ids: Indices of environments to resample.
    """
    num_resamples = len(env_ids)
    
    # Sample velocity commands from uniform distributions.
    # Note: Reusing tensor 'r' for efficiency (in-place uniform_ operation).
    r = torch.empty(num_resamples, device=self.device)
    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    
    # Sample heading target and assign heading control mode.
    if self.cfg.heading_command:
      assert self.cfg.ranges.heading is not None
      self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
      self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
    
    # Assign standing mode.
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    # Domain randomization: Initialize velocities for some environments.
    # This helps the policy adapt to various initial velocity conditions.
    init_vel_mask = r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob
    init_vel_env_ids = env_ids[init_vel_mask]
    
    if len(init_vel_env_ids) > 0:
      # Get current root state.
      root_pos = self.robot.data.root_link_pos_w[init_vel_env_ids]
      root_quat = self.robot.data.root_link_quat_w[init_vel_env_ids]
      
      # Set linear velocity in body frame to command velocity (XY only).
      lin_vel_b = self.robot.data.root_link_lin_vel_b[init_vel_env_ids].clone()
      lin_vel_b[:, :2] = self.vel_command_b[init_vel_env_ids, :2]
      root_lin_vel_w = quat_apply(root_quat, lin_vel_b)
      
      # Set angular velocity in body frame to command velocity (Z only).
      root_ang_vel_b = self.robot.data.root_link_ang_vel_b[init_vel_env_ids].clone()
      root_ang_vel_b[:, 2] = self.vel_command_b[init_vel_env_ids, 2]
      
      # Write updated state to simulation.
      root_state = torch.cat(
        [root_pos, root_quat, root_lin_vel_w, root_ang_vel_b], dim=-1
      )
      self.robot.write_root_state_to_sim(root_state, init_vel_env_ids)

  def _update_command(self) -> None:
    """Update velocity commands based on heading control and standing modes.
    
    Optimized version: Only compute heading errors for environments that need it.
    """
    # Heading control: Convert heading error to angular velocity command.
    if self.cfg.heading_command:
      # Filter environments that use heading control (efficiency optimization).
      heading_env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      
      if len(heading_env_ids) > 0:
        # Compute heading error only for selected environments.
        heading_error_selected = wrap_to_pi(
          self.heading_target[heading_env_ids] - self.robot.data.heading_w[heading_env_ids]
        )
        
        # Update global heading_error buffer for monitoring/debugging.
        self.heading_error[heading_env_ids] = heading_error_selected
        
        # Apply proportional control with saturation.
        self.vel_command_b[heading_env_ids, 2] = torch.clip(
          self.cfg.heading_control_stiffness * heading_error_selected,
          min=self.cfg.ranges.ang_vel_z[0],
          max=self.cfg.ranges.ang_vel_z[1],
        )
    
    # Standing mode: Zero all velocity commands.
    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    if len(standing_env_ids) > 0:
      self.vel_command_b[standing_env_ids, :] = 0.0

  # Visualization.

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw velocity command and actual velocity arrows.

    Visualizes the selected environment (visualizer.env_idx) with:
    - Blue arrow: Commanded linear velocity
    - Green arrow: Commanded angular velocity
    - Cyan arrow: Actual linear velocity
    - Light green arrow: Actual angular velocity

    Args:
        visualizer: Debug visualizer interface.
    """
    batch = visualizer.env_idx

    # Boundary check.
    if batch >= self.num_envs:
      return

    # Extract data for selected environment.
    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

    base_pos_w = base_pos_ws[batch]
    base_mat_w = base_mat_ws[batch]
    cmd = cmds[batch]
    lin_vel_b = lin_vel_bs[batch]
    ang_vel_b = ang_vel_bs[batch]

    # Skip if robot appears uninitialized (at origin).
    if np.linalg.norm(base_pos_w) < 1e-6:
      return

    # Helper to transform local coordinates to world coordinates.
    def local_to_world(
      vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
    ) -> np.ndarray:
      return pos + mat @ vec

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    # --- Command velocities ---
    
    # Command linear velocity arrow (blue).
    cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
    cmd_lin_to = local_to_world(
      (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
    )
    visualizer.add_arrow(
      cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
    )

    # Command angular velocity arrow (green).
    cmd_ang_from = cmd_lin_from
    cmd_ang_to = local_to_world(
      (np.array([0, 0, z_offset]) + np.array([0, 0, cmd[2]])) * scale
    )
    visualizer.add_arrow(
      cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
    )

    # --- Actual velocities ---
    
    # Actual linear velocity arrow (cyan).
    act_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
    act_lin_to = local_to_world(
      (np.array([0, 0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])) * scale
    )
    visualizer.add_arrow(
      act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
    )

    # Actual angular velocity arrow (light green).
    act_ang_from = act_lin_from
    act_ang_to = local_to_world(
      (np.array([0, 0, z_offset]) + np.array([0, 0, ang_vel_b[2]])) * scale
    )
    visualizer.add_arrow(
      act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
    )


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(CommandTermCfg):
  """Configuration for uniform velocity command generation.
  
  Attributes:
      asset_name: Name of robot entity in the scene.
      heading_command: Enable heading control mode.
      heading_control_stiffness: Proportional gain for heading error to angular velocity.
      rel_standing_envs: Fraction of environments in standing mode [0, 1].
      rel_heading_envs: Fraction of environments using heading control [0, 1].
      init_velocity_prob: Probability of initializing robot velocity to command [0, 1].
      ranges: Velocity sampling ranges.
      viz: Visualization settings.
  """
  
  asset_name: str
  heading_command: bool = False
  heading_control_stiffness: float = 1.0
  rel_standing_envs: float = 0.0
  rel_heading_envs: float = 1.0
  init_velocity_prob: float = 0.0
  class_type: type[CommandTerm] = UniformVelocityCommand

  @dataclass
  class Ranges:
    """Velocity command sampling ranges."""
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    heading: tuple[float, float] | None = None

  ranges: Ranges

  @dataclass
  class VizCfg:
    """Visualization configuration."""
    z_offset: float = 0.2  # Vertical offset for arrows [m].
    scale: float = 0.5     # Arrow length scaling factor.

  viz: VizCfg = field(default_factory=VizCfg)

  def __post_init__(self):
    """Validate configuration after initialization."""
    if self.heading_command and self.ranges.heading is None:
      raise ValueError(
        "The velocity command has heading commands active (heading_command=True) but "
        "the `ranges.heading` parameter is set to None."
      )
