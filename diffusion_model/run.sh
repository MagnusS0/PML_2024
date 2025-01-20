#!/bin/bash

# SDE experiments
echo "Running SDE experiments..."

# Default SDE
# python diffusion_model/train.py --model_type SDE

# DDPM experiments
echo "Running DDPM experiments..."

python diffusion_model/train.py --model_type DDPM --beta_schedule cosine --loss_type IS
# python diffusion_model/train.py --model_type DDPM --beta_schedule linear --loss_type IS

# python diffusion_model/train.py --model_type DDPM --beta_schedule cosine --loss_type LDS
# python diffusion_model/train.py --model_type DDPM --beta_schedule linear --loss_type LDS

# python diffusion_model/train.py --model_type DDPM --beta_schedule cosine --loss_type simple
# python diffusion_model/train.py --model_type DDPM --beta_schedule linear --loss_type simple

# python diffusion_model/train.py --model_type DDPM --beta_schedule cosine --loss_type LDS_2
# python diffusion_model/train.py --model_type DDPM --beta_schedule linear --loss_type LDS_2
