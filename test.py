from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from utils.downsample_wrapper import wrap_deepmind

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = wrap_deepmind(env)
env = JoypadSpace(env, COMPLEX_MOVEMENT)


done = True
for step in range(500000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    print(state.shape)
    env.render()

env.close()
