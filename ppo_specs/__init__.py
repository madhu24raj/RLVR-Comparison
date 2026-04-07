"""
PPO implementation for RLVR experiments (E2.7, E2.8).

Modules
-------
config      – PPOConfig dataclass and preset experiment configs
critic      – Value-function architectures: none / small / medium / large
advantage   – Advantage estimation and Monte Carlo error measurement
ppo_trainer – PPOTrainer (rollout generation, PPO-clip update, evaluation)
run_e2_7    – Head-to-head comparison experiment (PPO portion)
run_e2_8    – Critic quality sweep experiment
"""
