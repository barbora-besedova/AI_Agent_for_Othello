#!/bin/bash
set -e
source /home/cube/projects/AI_Agent_for_Othello/.venv/bin/activate
cd /home/cube/projects/AI_Agent_for_Othello/all_in_clean
PYTHONUNBUFFERED=1 python advanced_training.py   --load_model_path /home/cube/projects/AI_Agent_for_Othello/all_in_clean/models/guided_per_dqn_6_best_overnight.pth   --base_dir experiments   2>&1 | tee /home/cube/projects/AI_Agent_for_Othello/all_in_clean/training_experiment.log
