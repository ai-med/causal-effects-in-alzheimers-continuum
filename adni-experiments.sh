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
OUTDIR="/workspace/outputs/adni"
mkdir -p "${OUTDIR:?}/models" "${OUTDIR:?}/results" "${OUTDIR:?}/subst_conf"

. /opt/conda/bin/activate jupyter

set -xe

python -m causalad.adni.prepare_data \
    --adni_csv "${DATADIR:?}/adni-data.csv" \
    --outputs_dir "${OUTDIR:?}"

for m in "bpmf" "ppca"
do
    python -m causalad.adni.substitute_confounder deconf \
        --input_data "${OUTDIR:?}/adni_data_t.h5" \
        --model "${m}" \
        --latent_dim "6" \
        --output_file "${OUTDIR:?}/subst_conf/adni_${m}_subst_conf_dim6.h5"
done

for m in "original_model" "age_residualized_model" "combat_residualized_model"
do
    python -m causalad.adni.fit "${m}" \
        --outcome_file "${OUTDIR:?}/adni_data_t.h5" \
        --outputs_dir "${OUTDIR:?}/models"
done

python -m causalad.adni.fit "subst_conf_models" \
    --features_dir "${OUTDIR:?}/subst_conf/" \
    --outcome_file "${OUTDIR:?}/adni_data_t.h5" \
    --outputs_dir "${OUTDIR:?}/models"

python -m causalad.adni.plot coefs \
    --models_dir "${OUTDIR:?}/models" \
    --outputs_dir "${OUTDIR:?}/results"

python -m causalad.adni.plot ace \
    --data_file "${OUTDIR:?}/adni_data_t.h5" \
    --models_dir "${OUTDIR:?}/models" \
    --subst_conf_dir "${OUTDIR:?}/subst_conf/" \
    --outputs_dir "${OUTDIR:?}/results"
