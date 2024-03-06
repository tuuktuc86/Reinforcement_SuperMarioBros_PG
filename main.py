import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import random, datetime
from pathlib import Path
import numpy as np
import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace
import torch.optim as optim
#from finish_episode import *
from metrics import MetricLogger
from agent import Mario
from wrappers import ResizeObservation, SkipFrame
import torch
from tensorboardX import SummaryWriter
log_dir = "./save"  # 로그를 저장할 디렉토리를 지정하세요.
writer = SummaryWriter(log_dir=log_dir)

# Initialize Super Mario environment
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

# Limit the action-space to
#   0. walk right
#   1. jump right
env = JoypadSpace(
    env,
    [['right'],
    ['right', 'A']]
)

# Apply Wrappers to environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=4)

env.reset()

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

checkpoint = None # Path('checkpoints/2020-10-21T18-25-27/mario.chkpt')
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)

logger = MetricLogger(save_dir)

episodes = 40000

log_dir = "./save"  # 로그를 저장할 디렉토리를 지정하세요.
writer = SummaryWriter(log_dir=log_dir)

model = torch.load("save_dir/mario_net_31500.chkpt")
#print(model)
#policy gradient discount factor
gamma = 0.99
#이건 무슨 용도인지 잘 모르겠다.
eps = np.finfo(np.float32).eps.item()

def finish_episode():
    #print("in finish episode")
    R = 0
    policy_loss = []
    rewards = []

    for r in mario.rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)

    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(mario.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    mario.optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    logger.policyLoss = policy_loss
    policy_loss.backward()
    mario.optimizer.step()
    del mario.rewards[:]
    del mario.saved_log_probs[:]


### for Loop that train the model num_episodes times by playing the game
for e in range(episodes):

    state = env.reset()

    # Play the game!
    while True:
        #print("befor3")
        # 3. Show environment (the visual) [WIP]
        env.render()
        #print("befor4")
        # 4. Run agent on the state

        state = np.array(state)
        state = torch.FloatTensor(state)
        state = state.unsqueeze(0)
        # print(state)
        action = model(state)

        action = int(action)
        #print("action = ", action)
        #print("befor5")
        # 5. Agent performs action
        next_state, reward, done, info = env.step(action)
        #print("reward = ", reward)
        #policy gradient
        mario.rewards.append(reward)

        #print("reward = ", reward)
        #print("befor6")
        # 6. Remember
        #mario.cache(state, next_state, action, reward, done)
        #print("befor7")
        # 7. Learn
        #q, loss = mario.learn()
        #print("befor8")
        # 8. Logging
        logger.log_step(reward)

        # 9. Update state
        state = next_state

        # 10. Check if end of game
        if done or info['flag_get']:
            #finish_episode()
            logger.log_episode()
            #print("break")
            break



    if e % 20 == 0:

        # writer.add_scalar("Reward", reward, e)
        # writer.add_scalar("Loss", loss, e)
        logger.record(
            episode=e,
            step=mario.curr_step
        )
    # if e%500 == 0:
    #     save_path = f"save_dir/mario_net_{int(e)}.chkpt"
    #     torch.save(
    #         dict(
    #             model=mario.net.state_dict(),
    #             # exploration_rate=self.exploration_rate
    #         ),
    #         save_path
    #     )
    #     print(f"MarioNet saved to save_dir at step {e}")