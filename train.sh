#! /bin/bash

set -x

echo "Training TDMPC"

./isaaclab.sh -p scripts/tdmpc/train.py --task Isaac-Quadcopter-Direct-v1 --num_envs 1 --video --use_wandb --exp_name "tdmpc_quadcopter_v1" --save_model

# ./isaaclab.sh -p scripts/tdmpc/train.py --task Isaac-Quadcopter-Direct-v2 --num_envs 1 --video --use_wandb --exp_name "tdmpc_quadcopter_v2" --save_model