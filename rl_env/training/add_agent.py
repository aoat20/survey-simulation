import wandb


sweep_id = 'rxgk7rlr'

wandb.agent(sweep_id=sweep_id,project='surrey',entity='eprc20')