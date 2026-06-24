#!/bin/bash
set -e
source /home/cube/projects/AI_Agent_for_Othello/.venv/bin/activate
cd /home/cube/projects/AI_Agent_for_Othello/all_in_clean

PYTHONUNBUFFERED=1 python train_vs_minimax.py   --use_per --heuristic_weight 0.2   --load_model_path /home/cube/projects/AI_Agent_for_Othello/all_in_clean/models/against_minimax_overnight/guided_per_dqn_minimax_30000.pth   --model_path /home/cube/projects/AI_Agent_for_Othello/all_in_clean/models/against_minimax_overnight/guided_per_dqn_minimax_30000.pth   --num_episodes 30000 --save_every 500 --eval_every 500   --record_game_eps 5000,10000,20000,30000   --max_minutes 148   --epsilon_start 0.05 --epsilon_end 0.01 --epsilon_decay 0.9995   2>&1 | tee /home/cube/projects/AI_Agent_for_Othello/all_in_clean/training_overnight_resumed.log
