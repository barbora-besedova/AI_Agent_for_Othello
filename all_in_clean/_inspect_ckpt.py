import torch
ckpt = torch.load('Training/experiments_002/model_checkpoints/model_ep1000.pth', map_location='cpu', weights_only=True)
print('board_size:', ckpt.get('board_size'))
print('keys:', list(ckpt.keys()))
print('config:', ckpt.get('config'))
print('train_steps:', ckpt.get('train_steps'))
