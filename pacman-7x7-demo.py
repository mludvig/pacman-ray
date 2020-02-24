#!/usr/bin/env python3

import os
import sys

import gym
from gym.spaces import Discrete, Box
from gym import wrappers

from gym_pacman.PacMan_v0 import PacMan_v0

import ray
from ray.rllib.agents import ppo

CHECKPOINT_FILE="checkpoint-7x7-trained/checkpoint-12700"

if len(sys.argv) > 1:
    path = sys.argv[1]
elif os.access(CHECKPOINT_FILE, os.R_OK):
    path = CHECKPOINT_FILE
else:
    path = ""

if path:
    print(f"Will attempt to restore checkpoint: {path}")

ray.init()

env_config = {
    "board_size": (7, 7),
}

print("== Training ==")

trainer = ppo.PPOTrainer(env=PacMan_v0, config={
    "env_config": env_config,
})

if path:
    trainer.restore(path)
    print(f"Restored checkpoint {path}")
else:
    trainer.train()
    path = trainer.save()
    print(f"Saved checkpoint {path}")

print("== Policy / Model ==")
policy = trainer.get_policy()
policy.model.base_model.summary()

print("== Evaluating ==")

env = PacMan_v0(config=env_config)
#env = wrappers.Monitor(env, "/tmp/render", force=True)

for _ in range(10):
    observation = env.reset()

    done = False
    while not done:
        env.render()
        action, info1, info2 = trainer.compute_action(observation, full_fetch=True)
        #print(f"observation: {observation} -> action: {action} / {info1} / {info2}")
        observation, reward, done, _ = env.step(action)
    env.render()
    #print(f"observation: {observation} -> done")
env.close()
