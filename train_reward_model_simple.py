
from trainer_reward import ProbRewardModelTrainer
import yaml

from tqdm import tqdm
import tiktoken
import torch

with open('config_reward.yaml') as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)
    # nested dictionary structure
    config = {}               
    for k, v in conf.items():
        for k2, v2 in v.items():
            config[k2] = v2
print(config)

trainer = ProbRewardModelTrainer(config, prob_reward=True)

trainer.train()

