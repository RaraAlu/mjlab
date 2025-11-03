#!/bin/bash
# MUJOCO_GL=egl uv run train Mjlab-Velocity-Flat-Unitree-G1 \
#   --env.scene.num-envs 4096


MUJOCO_GL=egl uv run train Mjlab-Velocity-Flat-Agibot-X1 \
  --env.scene.num-envs 8192
