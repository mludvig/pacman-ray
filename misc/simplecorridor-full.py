#!/usr/bin/env python3

import gym
from gym.spaces import Discrete, Box

import ray
from ray.rllib.agents import ppo

class SimpleCorridor(gym.Env):
    def __init__(self, config):
        self.end_pos = config.get("corridor_length", 5)
        self.cur_pos = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(0.0, self.end_pos, shape=(1, ))

    def reset(self):
        self.cur_pos = 0
        return [self.cur_pos]

    def step(self, action):
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        elif action == 1:
            self.cur_pos += 1
        done = self.cur_pos >= self.end_pos
        return [self.cur_pos], 1 if done else 0, done, {}


ray.init()

print("== Training ==")

trainer = ppo.PPOTrainer(env=SimpleCorridor)
trainer.train()

print("== Evaluating ==")

env = SimpleCorridor(config={})
observation = env.reset()

done = False
while not done:
    action, info1, info2 = trainer.compute_action(observation, full_fetch=True)
    print(f"observation: {observation} -> action: {action} / {info1} / {info2}")
    observation, reward, done, _ = env.step(action)
print(f"observation: {observation} -> done")
