#!/usr/bin/env python3

import pprint

import gym
import gym_pacman

import ray
from ray.rllib.agents import ppo, dqn, impala, a3c

ray.init()

trainer_class = ppo.PPOTrainer
#trainer_class = dqn.DQNTrainer
#trainer_class = a3c.A3CTrainer
#trainer_class = impala.ImpalaTrainer

trainer = trainer_class(env=gym_pacman.PacMan_v0, config={
    "log_level": "INFO",
    #"eager": True,     # disables tensorboard stats
    "lr": 5e-4,
    "gamma": 0.6,
    "num_workers": 4,
    "env_config": {  # config to pass to env class
        "board_size": (7, 7),
        "max_moves": 100,
    },
    "vf_share_layers": True, # LSTM needs vf_share_layers=True
    "model": {
        "use_lstm": True,
    },
})

iter = 0
while True:
    iter += 1
    print(f"=== Training iter: {iter} ===")
    trainer_output = trainer.train()
    pprint.pprint(trainer_output)
    if iter % 100 == 0:
        checkpoint = trainer.save()
        print(f"Checkpoint: {checkpoint}")
    if trainer_output['episodes_total'] >= 200000:
        break
trainer.save()
print(f"Final checkpoint: {checkpoint}")
