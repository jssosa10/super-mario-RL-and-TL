from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from utils.downsample_wrapper import wrap_deepmind
from PIL import Image
import os

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v1')
env = wrap_deepmind(env)
env = JoypadSpace(env, COMPLEX_MOVEMENT)

done = True
os.chdir("data/1-1")
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    state = state.transpose(2, 0, 1)
    state = state[0]
    img = Image.fromarray(state, 'L')
    im = Image.fromarray(state)
    if step % 10000 == 0:
        print("steps", step+1)
    img.save("step"+str(step)+".jpg")

env.close()
os.chdir("../..")
env = gym_super_mario_bros.make('SuperMarioBros-1-2-v1')
env = wrap_deepmind(env)
env = JoypadSpace(env, COMPLEX_MOVEMENT)

done = True
os.chdir("data/1-2")
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    state = state.transpose(2, 0, 1)
    state = state[0]
    img = Image.fromarray(state, 'L')
    im = Image.fromarray(state)
    if step % 10000 == 0:
        print("steps", step+1)
    img.save("step"+str(step)+".jpg")

env.close()
