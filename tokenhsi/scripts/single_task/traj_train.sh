#!/bin/bash

python ./tokenhsi/run.py --task HumanoidTraj \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_wandb.yaml \
    --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_traj_wandb.yaml \
    --motion_file tokenhsi/data/dataset_amass_loco/dataset_amass_loco.yaml \
    --num_envs 1024 \
    --headless