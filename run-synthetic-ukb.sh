#!/bin/bash

set -xeu

for m in "ppca" "bpmf"
do
    python -m causalad.synthetic_experiments deconf --model $m --latent_dim 5 --output_dir data/outputs/synthetic_ukb
done

python -m causalad.synthetic_experiments fit --latent_dim 5 --output_dir data/outputs/synthetic_ukb --n_repeat 1000 --n_jobs 10

python -m causalad.synthetic_experiments evaluate --latent_dim 5 --output_dir data/outputs/synthetic_ukb
