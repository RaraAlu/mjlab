"""Agibot X1 constants."""

from pathlib import Path
import math

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
)
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import ActuatorCfg, CollisionCfg

##
# MJCF and assets.
##

X1_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "agibot_x1" / "xmls" / "x1.xml"
)
assert X1_XML.exists()

def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, X1_XML.parent / "assets", meshdir)
  return assets

def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(X1_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec

##
# Actuator config (Zhiyuan PowerFlow based).
##

# Conversion factor from RPM to Rad/s
RPM_TO_RADS = 2.0 * math.pi / 60.0

# General Servo Model Parameters (Used for LQR control model)
NATURAL_FREQ = 10 * 2.0 * math.pi # 10Hz natural frequency
DAMPING_RATIO = 2.0

# --- 1. R86-3 (Type A): High-Torque Legs ---
# Used for 6 joints: Major Leg Joints (Hip Pitch, Knee Pitch, Ankle Pitch)
# Peak Torque: 200 Nm, Peak Speed: 85 rpm (8.90 rad/s)
ARMATURE_R86_3 = 0.075 # ESTIMATE
VELOCITY_LIMIT_R86_3 = 85.0 * RPM_TO_RADS
EFFORT_LIMIT_R86_3 = 200.0

ACTUATOR_R86_3 = ElectricActuator(
  reflected_inertia=ARMATURE_R86_3,
  velocity_limit=VELOCITY_LIMIT_R86_3,
  effort_limit=EFFORT_LIMIT_R86_3,
)
STIFFNESS_R86_3 = ACTUATOR_R86_3.reflected_inertia * NATURAL_FREQ**2
DAMPING_R86_3 = 2.0 * DAMPING_RATIO * ACTUATOR_R86_3.reflected_inertia * NATURAL_FREQ

X1_ACTUATOR_R86_3 = ActuatorCfg(
  joint_names_expr=[
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_knee_pitch_joint",

  ],
  effort_limit=ACTUATOR_R86_3.effort_limit,
  armature=ACTUATOR_R86_3.reflected_inertia,
  stiffness=STIFFNESS_R86_3,
  damping=DAMPING_R86_3,
)

# --- 2. R86-2 (Type B): Medium-Torque Legs/Torso ---
# Used for 9 joints: Torso (3), Secondary Leg Joints (Hip Roll/Yaw, Ankle Roll)
# Peak Torque: 80 Nm, Peak Speed: 260 rpm (27.23 rad/s)
ARMATURE_R86_2 = 0.015 # ESTIMATE
VELOCITY_LIMIT_R86_2 = 260.0 * RPM_TO_RADS
EFFORT_LIMIT_R86_2 = 80.0

ACTUATOR_R86_2 = ElectricActuator(
  reflected_inertia=ARMATURE_R86_2,
  velocity_limit=VELOCITY_LIMIT_R86_2,
  effort_limit=EFFORT_LIMIT_R86_2,
)
STIFFNESS_R86_2 = ACTUATOR_R86_2.reflected_inertia * NATURAL_FREQ**2
DAMPING_R86_2 = 2.0 * DAMPING_RATIO * ACTUATOR_R86_2.reflected_inertia * NATURAL_FREQ

X1_ACTUATOR_R86_2 = ActuatorCfg(
  joint_names_expr=[
    # Arms
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    # Legs
    ".*_hip_yaw_joint",
    # Torso (3 joints)
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
  ],
  effort_limit=ACTUATOR_R86_2.effort_limit,
  armature=ACTUATOR_R86_2.reflected_inertia,
  stiffness=STIFFNESS_R86_2,
  damping=DAMPING_R86_2,
)

# --- 3. R52 (Type C): Low-Torque Arms/Upper Body ---
# Used for 10 joints: All Arm Joints (5 per arm)
# Peak Torque: 19 Nm, Peak Speed: 130 rpm (13.61 rad/s)
ARMATURE_R52 = 0.0035 # ESTIMATE
VELOCITY_LIMIT_R52 = 130.0 * RPM_TO_RADS
EFFORT_LIMIT_R52 = 19.0 

ACTUATOR_R52 = ElectricActuator(
  reflected_inertia=ARMATURE_R52,
  velocity_limit=VELOCITY_LIMIT_R52,
  effort_limit=EFFORT_LIMIT_R52,
)
STIFFNESS_R52 = ACTUATOR_R52.reflected_inertia * NATURAL_FREQ**2
DAMPING_R52 = 2.0 * DAMPING_RATIO * ACTUATOR_R52.reflected_inertia * NATURAL_FREQ

X1_ACTUATOR_R52 = ActuatorCfg(
  joint_names_expr=[
    # Arms
    ".*_shoulder_yaw_joint",
    ".*_elbow_pitch_joint",
    ".*_elbow_yaw_joint",
    # Legs
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
  ],
  effort_limit=ACTUATOR_R52.effort_limit,
  armature=ACTUATOR_R52.reflected_inertia,
  stiffness=STIFFNESS_R52,
  damping=DAMPING_R52,
)

# Note on L28: The L28 linear actuators are assumed to control the grippers (not modeled as rotary joints here)
# or internal mechanisms not directly mapped to the 25 DoF kinematic chain in x1.xml.

##
# Keyframe config.
##

# Stand Upright (Straight legs, slight knee bend, arms relaxed)
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.65), # Assuming a standing height of 1.0m (X1 is 130cm tall, so COM is lower)
  joint_pos={
    # Core
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,

    # Legs (Deeper squat)
    ".*_hip_pitch_joint": 0.0,
    ".*_hip_roll_joint": 0.0,
    ".*_hip_yaw_joint": 0.0,
    ".*_knee_pitch_joint": 0.0,
    ".*_ankle_pitch_joint": 0.0,
    ".*_ankle_roll_joint": 0.0,

     # Arms (Tucked in)
    ".*_shoulder_pitch_joint": 0.16,
    ".*_shoulder_roll_joint": -0.1,
    ".*_shoulder_yaw_joint": 0.0,
    ".*_elbow_pitch_joint": 0.3,
    ".*_elbow_yaw_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

# Ready / Slightly Squatted Pose
READY_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.63), # Lowered height
  joint_pos={
    # Core
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,

    # Legs (Deeper squat)
    ".*_hip_pitch_joint": 0.0,
    ".*_hip_roll_joint": 0.0,
    ".*_hip_yaw_joint": 0.0,
    ".*_knee_pitch_joint": 0.0,
    ".*_ankle_pitch_joint": 0.0,
    ".*_ankle_roll_joint": 0.0,

     # Arms (Tucked in)
    ".*_shoulder_pitch_joint": 0.16,
    ".*_shoulder_roll_joint": -0.1,
    ".*_shoulder_yaw_joint": 0.0,
    ".*_elbow_pitch_joint": 0.3,
    ".*_elbow_yaw_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3 and custom friction and solimp.
FOOT_GEOM_NAMES_REGEX = r"^(left|right)_foot[1-7]_collision$"
FOOT_FRICTION = 0.6 # Standard MuJoCo friction coefficient

FULL_COLLISION = CollisionCfg(
  geom_names_expr=[".*_collision"],
  # Feet get condim=3 (3D friction)
  condim={FOOT_GEOM_NAMES_REGEX: 3, ".*_collision": 1},
  priority={FOOT_GEOM_NAMES_REGEX: 1},
  friction={FOOT_GEOM_NAMES_REGEX: (FOOT_FRICTION,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=[".*_collision"],
  contype=0,
  conaffinity=1,
  condim={FOOT_GEOM_NAMES_REGEX: 3, ".*_collision": 1},
  priority={FOOT_GEOM_NAMES_REGEX: 1},
  friction={FOOT_GEOM_NAMES_REGEX: (FOOT_FRICTION,)},
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=[FOOT_GEOM_NAMES_REGEX],
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(FOOT_FRICTION,),
)

##
# Final config.
##

X1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    X1_ACTUATOR_R86_3,
    X1_ACTUATOR_R86_2,
    X1_ACTUATOR_R52,
  ),
  soft_joint_pos_limit_factor=0.9,
)

X1_ROBOT_CFG = EntityCfg(
  init_state=READY_KEYFRAME, # Using READY pose for base configuration
  collisions=(FULL_COLLISION,),
  spec_fn=get_spec,
  articulation=X1_ARTICULATION,
)

# Calculate action scale (essential for normalized action space in RL)
X1_ACTION_SCALE: dict[str, float] = {}
for a in X1_ARTICULATION.actuators:
  e = a.effort_limit
  s = a.stiffness
  names = a.joint_names_expr
  # Assuming effort_limit and stiffness are float constants for now
  if not isinstance(e, dict):
    e = {n: e for n in names}
  if not isinstance(s, dict):
    s = {n: s for n in names}
  for n in names:
    if n in e and n in s and s[n]:
      X1_ACTION_SCALE[n] = 0.25 * e[n] / s[n]

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(X1_ROBOT_CFG)
  
  # Launch the viewer to check configuration
  viewer.launch(robot.spec.compile())