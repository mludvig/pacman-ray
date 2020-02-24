#!/usr/bin/env python3

import gym

import ray
from ray.rllib.agents import ppo

ray.init()

print("== Training ==")

trainer = ppo.PPOTrainer(env='CartPole-v0')
trainer.train()

print("== Policy / Model ==")
policy = trainer.get_policy()
policy.model.base_model.summary()

print("== Evaluating ==")

env = gym.make('CartPole-v0')
observation = env.reset()
print(f"observation: {observation}")

done = False
while not done:
    env.render()
    action = trainer.compute_action(observation)
    observation, reward, done, _ = env.step(action)
    print(f"action: {action}")
