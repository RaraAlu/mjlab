#!/bin/bash
uv run python -m mjlab.scripts.play_joystick Mjlab-Velocity-Flat-Agibot-X1-Play \
    --checkpoint-file ~/Work/mjlab/logs/rsl_rl/x1_velocity/2025-10-20_09-21-07/model_29999.pt \
    --num-envs 1 \
    --agent joystick \
    # --viewer viser
# uv run python -m mjlab.scripts.play_joystick Mjlab-Velocity-Flat-Unitree-G1-Play \
#     --checkpoint-file ~/Work/mjlab/logs/rsl_rl/g1_velocity/2025-11-03_16-36-06/model_2000.pt \
#     --num-envs 1 \
#     --agent joystick \


