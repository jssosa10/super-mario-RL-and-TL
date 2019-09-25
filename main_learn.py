from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from utils.downsample_wrapper import wrap_deepmind
from utils.schedule import LinearSchedule
from gym import wrappers
from networks.DQN import DQN
from scripts.learn import dqn_learn, OptimizerSpec
import torch.optim as optim

import torch
import numpy as np
import random

SEED = 1
BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 100000
LEARNING_STARTS = 50000
LEARNING_FREQ = 4
TARGER_UPDATE_FREQ = 10000
LEARNING_RATE = 0.00025
ALPHA = 0.95
ALPHA_P = 0.6
EPS = 0.01

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

env = wrap_deepmind(env)
env = JoypadSpace(env, COMPLEX_MOVEMENT)
expt_dir = 'Game_video'
env = wrappers.Monitor(env, expt_dir, force=True, video_callable=lambda episode_id: True)

optimizer_spec = OptimizerSpec(
    constructor=optim.RMSprop,
    kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
)

exploration_schedule = LinearSchedule(2000000, 0.1, 0.9)
annelation_schedule = LinearSchedule(2000000, 1.0, 0.4)

dqn_learn(
    env=env,
    q_func=DQN,
    optimizer_spec=optimizer_spec,
    exploration=exploration_schedule,
    replay_buffer_size=REPLAY_BUFFER_SIZE,
    batch_size=BATCH_SIZE,
    gamma=GAMMA,
    learning_starts=LEARNING_STARTS,
    learning_freq=LEARNING_FREQ,
    alpha=ALPHA_P,
    annelation=annelation_schedule,
    target_update_freq=TARGER_UPDATE_FREQ,
)
