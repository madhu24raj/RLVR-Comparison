Running and Logging:

CPU/single GPU:
python dpo_specs/run_e2_7.py --local-test

Full Cluster Run (Seed 42):
python dpo_specs/run_e2_7.py --seed 42

dpo_specs/run_e2_7.py utilizes the exact same eval.metrics.ExperimentLogger class

logs will populate in the /results/ directory as results/ppo_e2_7_seed42_DPO.json

TODO:
(can plot the DPO learning curve directly comparable to PPO and GRPO using the existing plot_convergence funcs)