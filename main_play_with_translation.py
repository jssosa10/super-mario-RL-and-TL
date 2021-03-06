from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from utils.downsample_wrapper import wrap_deepmind
from utils.image2image_wrapper import wrapImage
from gym import wrappers
from networks.DQN import DQN
from scripts.play import dqn_play


import torch
import numpy as np
import random

SEED = 1

env = gym_super_mario_bros.make('SuperMarioBros-1-2-v1')
env.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

env = wrap_deepmind(env)
env = wrapImage(env)
env = JoypadSpace(env, COMPLEX_MOVEMENT)
expt_dir = 'Game_play3'
env = wrappers.Monitor(env, expt_dir, force=True, video_callable=lambda episode_id: True)
dqn_play(
    env=env,
    q_func=DQN,
    with_translation=False
)
