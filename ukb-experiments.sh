#!/bin/bash
# This file is part of Estimation of Causal Effects in the Alzheimer's Continuum (Causal-AD).
#
# Causal-AD is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Causal-AD is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Causal-AD. If not, see <https://www.gnu.org/licenses/>.

DATADIR="/workspace/data"
OUTDIR="/workspace/outputs/ukb"
mkdir -p "${OUTDIR:?}/subst_conf/" "${OUTDIR:?}/simulations/" "${OUTDIR:?}/results/"

. /opt/conda/bin/activate jupyter

set -xe

python -m causalad.ukb.prepare_data \
    --ukb_csv "${DATADIR:?}/ukb-data.csv" \
    --outputs_dir "${OUTDIR:?}"

python -m causalad.ukb.simulate_data \
    --data_path "${OUTDIR:?}/ukb_data_t.h5" \
    --n_repeats "100" \
    --output_dir "${OUTDIR:?}/simulations/"

for m in "bpmf" "ppca"
do
for ldim in $(seq 8)
do
    python -m causalad.ukb.substitute_confounder deconf \
        -i "${OUTDIR:?}/ukb_data_t.h5" \
        -m "${m}" \
        --latent_dim "${ldim}" \
        -o "${OUTDIR:?}/subst_conf/subst_conf_${m}-${ldim}_data.h5"
done
done

for m in "no_confounders" "observed_confounders" "oracle_model"
do
	python -m causalad.ukb.fit ${m} \
		--data_file "${OUTDIR:?}/ukb_data_t.h5" \
		--models_dir "${OUTDIR:?}/simulations/"
done

python -m causalad.ukb.fit subst_conf_models \
    --features_dir "${OUTDIR:?}/subst_conf/" \
    --models_dir "${OUTDIR:?}/simulations/" \
    --data_file "outputs/ukb/ukb_data_t.h5"

python -m causalad.ukb.summarize \
    --models_dir "${OUTDIR:?}/simulations/" \
    --subst_conf_dir "${OUTDIR:?}/subst_conf/" \
    --outputs_dir "${OUTDIR:?}/results/"
