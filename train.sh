#! /bin/bash

set -x

echo "Training TDMPC"

# ./isaaclab.sh -p scripts/tdmpc/train.py --task Isaac-Quadcopter-Direct-v1 --num_envs 1 --headless --use_wandb --exp_name "tdmpc_quadcopter_v1" --save_model

./isaaclab.sh -p scripts/tdmpc/train.py --task Isaac-Quadcopter-Direct-v2 --num_envs 1 \
 --use_wandb --exp_name "tdmpc_quadcopter_v2_finetune_all_models" --save_model --headless \
 --ckpt_path "logs/tdmpc/quadcopter-landing/2025-04-23_18-26-29/models/model_140000.pt"

./isaaclab.sh -p scripts/tdmpc/train.py --task Isaac-Quadcopter-Direct-v2 --num_envs 1 \
 --use_wandb --exp_name "tdmpc_quadcopter_v2_finetune_dynamics_only" --save_model --headless \
 --ckpt_path "logs/tdmpc/quadcopter-landing/2025-04-23_18-26-29/models/model_140000.pt" \
 --finetune_dynamics