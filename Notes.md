# project notes: ppo vs grpo vs dpo

goal: comparing 3 llm alignment methods (ppo, grpo, dpo) on math/coding tasks using verifiable rewards (rlvr). 

### repo needs
- **models**: ppo (needs a modular critic), grpo, dpo - all built on llama-3 8b
- **data**: loaders for gsm8k and humaneval. need a pipeline to generate synthetic preference pairs for dpo
- **scripts**: main runners for exps 2.7, 2.8, and 2.9
- **eval/logging**: trackers for accuracy, training stability, convergence speed and advantage estimation error

### core experiments
- **exp 2.7 (head-to-head)**: run all 3 models on both datasets with matched compute - log everything across 3 seeds.
- **exp 2.8 (critic sweep)**: run ppo with different critic sizes (none, small, med, large) vs grpo. need to generate a crossover plot for critic error vs accuracy..
- **exp 2.9 (label regimes)**: compare grpo and dpo using full, sparse (10%), and noisy (10% flipped) label conditions.

tasks:
build the data loaders, implement the dpo synthetic pair pipeline, lead exp 2.9
build ppo and the varying critic networks, calculate monte carlo advantage error, lead exp 2.8.
build grpo, set up the central logging framework, lead exp 2.7.
