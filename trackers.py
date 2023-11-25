import os
import wandb 
os.environ["WANDB_API_KEY"] = '63b00c6db86b0cb5fd3146e0959cfc9392ca5a3f'


class Tracker:
    def __init__(self, args_dict):
        wandb.init(
            # set the wandb project where this run will be logged
            project="img2txt",
            
            # track hyperparameters and run metadata
            config=args_dict
        )


    def log(self, kwargs, step=None):
        if step is not None:
            wandb.log(kwargs, step=step)
        else:
            wandb.log(kwargs)