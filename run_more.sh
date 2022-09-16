python pretrain.py agent=cic domain=quadruped seed=1 use_wandb=true
python finetune.py task=quadruped_run snapshot_ts=2000000 obs_type=states agent=cic reward_free=false seed=1
python finetune.py task=quadruped_walk snapshot_ts=2000000 obs_type=states agent=cic reward_free=false seed=1
python finetune.py task=quadruped_stand snapshot_ts=2000000 obs_type=states agent=cic reward_free=false seed=1
python finetune.py task=quadruped_jump snapshot_ts=2000000 obs_type=states agent=cic reward_free=false seed=1

python pretrain.py agent=cic domain=quadruped seed=2 use_wandb=true
python finetune.py task=quadruped_run snapshot_ts=2000000 obs_type=states agent=cic reward_free=false seed=2
python finetune.py task=quadruped_walk snapshot_ts=2000000 obs_type=states agent=cic reward_free=false seed=2
python finetune.py task=quadruped_stand snapshot_ts=2000000 obs_type=states agent=cic reward_free=false seed=2
python finetune.py task=quadruped_jump snapshot_ts=2000000 obs_type=states agent=cic reward_free=false seed=2

python pretrain.py agent=cic domain=quadruped seed=3 use_wandb=true
python finetune.py task=quadruped_run snapshot_ts=2000000 obs_type=states agent=cic reward_free=false seed=3
python finetune.py task=quadruped_walk snapshot_ts=2000000 obs_type=states agent=cic reward_free=false seed=3
python finetune.py task=quadruped_stand snapshot_ts=2000000 obs_type=states agent=cic reward_free=false seed=3
python finetune.py task=quadruped_jump snapshot_ts=2000000 obs_type=states agent=cic reward_free=false seed=3