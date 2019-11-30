from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from utils.downsample_wrapper import wrap_deepmind
from utils.image2image_wrapper import wrapImage
from gym import wrappers

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v1')
env = wrap_deepmind(env)
env = wrapImage(env)
env = JoypadSpace(env, COMPLEX_MOVEMENT)
expt_dir = 'Game_video'
env = wrappers.Monitor(env, expt_dir, force=True, video_callable=lambda episode_id: True)


done = True
for step in range(3):
    if done:
        state = env.reset()
    # print(env.action_space.n)
    state, reward, done, info = env.step(env.action_space.sample())
    print(state)
    env.render()

env.close()
