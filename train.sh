#!/bin/bash
MUJOCO_GL=egl uv run train Mjlab-Velocity-Flat-Agibot-X1 \
  --env.scene.num-envs 4096
