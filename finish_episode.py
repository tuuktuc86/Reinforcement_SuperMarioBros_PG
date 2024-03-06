import torch


from main import *
def finish_episode():
    print("in finish episode")
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
    policy_loss.backward()
    mario.optimizer.step()
    del mario.rewards[:]
    del mario.saved_log_probs[:]