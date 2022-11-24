# Estimation of Causal Effects in the Presence of Unobserved Confounding in the Alzheimer's Continuum

[![Paper](https://img.shields.io/static/v1?label=DOI&message=10.1002%2falz.12825&color=3a7ebb)](https://doi.org/10.1002/alz.12825)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

This repository contains the code to the paper "Identification of causal effects of neuroanatomy on cognitive decline requires modeling unobserved confounders."
If you are using this code, please cite:

```
@article(Poelsterl2022-adj,
  title   = {{Identification of causal effects of neuroanatomy on cognitive decline requires modeling unobserved confounders}},
  author  = {P{\"{o}}lsterl, Sebastian and Wachinger, Christian},
  journal = {Alzheimer's Dement},
  year    = {2022},
  pages   = {},
  doi     = {10.1002/alz.12825},
}
```

## Requirements

It is recommended to run the code via [Docker](https://docs.docker.com/install/).

If you want to use the code for development, you can use [conda](https://conda.io/miniconda.html)
to create an environment with all dependencies from `requirements.yaml`.

## Building the Docker Image

Pre-built packages are provided, but you can also build the docker image yourself:

1. Install [Docker](https://docs.docker.com/install/).
2. Build Docker image `causalad`:
```bash
docker build -t causalad .
```

## Data

This section provides an overview on how to obtain the data to
reproduce the results presented in the paper. As data cannot
be shared publicly, you will have to perform the data processing
yourself to fill in the missing values of `data/adni-data-template.csv`
and `data/ukb-data-template.csv`. These files list the patient ID,
and visit and image ID for ADNI, which uniquely identify the data
you need to obtain. We expect that you have been approved to access
the data and are familiar with the data portals of ADNI and UK Biobank.

### Alzheimer’s Disease Neuroimaging Initiative (ADNI)

1. Log in to the [ADNI Data Portal](http://adni.loni.usc.edu/).
2. Download `ADNIMERGE.CSV` and `UPENNBIOMK_MASTER.csv`.
3. Use `ABETA`, `PTAU`, `TAU` from `UPENNBIOMK_MASTER.csv` to determine
   which patients have an Alzheimer's pathologic by
  creating a column `ATN_status` that describes the A/T/N scheme, e.g.
  `A+/T+/N-` if `ABETA ≤ 192`, `PTAU ≥ 23`, and `TAU < 93`, following the
  thresholds from [Ekman et al., 2018](https://doi.org/10.1038/s41598-018-26151-8):
  > The individual CSF values were considered pathological (+)
  > if ≤192 pg/ml for Aβ42, ≥93 pg/ml for t-tau, and ≥23 pg/ml for p-tau.

4. Download T1 structural brain MRI from the [ADNI Data Portal](http://adni.loni.usc.edu/)
   and segment each with [FreeSurfer 5.3](https://www.freesurfer.net/) to obtain volume
   and thickness measurements.
5. Fill in the values of `data/adni-data-template.csv` by taking `ABETA`, `PTAU`, `TAU`
   from `UPENNBIOMK_MASTER.csv`, `ATN_status` from above, volume and thickness measurements
   computed by FreeSurfer, and the remaining variables from `ADNIMERGE.CSV`.
   Save the resulting file as `data/adni-data.csv`.

### UK Biobank (UKB)

1. Log in to the [UK Biobank Access Management System](https://bbams.ndph.ox.ac.uk/ams/resApplications).
2. Download data on [Sex](https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=31), and [Age](https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=21003)
 at first imaging visit.
3. Download T1 structural brain MRI and segment each with [FreeSurfer 5.3](https://www.freesurfer.net/) to obtain volume measurements.
4. Fill in the values of the `data/ukb-data-template.csv` and save the result as `data/ukb-data.csv`.


## Estimating Causal Effects

### Effect of Neuroanatomy on ADAS13

1. Make sure your created `data/adni-data.csv` as outlined above.
2. The entire workflow is summarized in a shell script, which can
   be executed by running:
```bash
docker run -it --rm \
-v $(pwd)/data:/workspace/data \
-v $(pwd)/outputs:/workspace/outputs \
ghcr.io/ai-med/causal-effects-in-alzheimers-continuum:v0.2.0 \
./adni-experiments.sh
```
3. Upon completion, the main results will be available in the `outputs/adni/results` folder.

   1. `plot-betareg-coef_outputs.ipynb`: This notebook will contain a figure comparing the
      estimated credible intervals for each model.
   2. `estimate_ace_outputs.ipynb`: This notebook will contain figures comparing the average
      causal effect (ACE) across models.

4. The estimated substitute confounders will be stored in the `outputs/adni/subst_conf` folder.

      1. `adni_bpmf_subst_conf_dim6.h5`: Transformed features with 6 substitute confounders estimated by BPMF.
      2. `adni_ppca_subst_conf_dim6.h5`: Transformed features with 6 substitute confounders estimated by PPCA.

5. The estimated mean coefficients for all models will be stored in the `outputs/adni/models` folder.

      1. `coef_adni_bpmf_subst_conf_dim6.csv`: Estimated coefficients of Beta-regression model when accounting for
         observed confounders and 6 substitute confounders estimated by BPMF.
      2. `coef_adni_ppca_subst_conf_dim6.csv`: Estimated coefficients of Beta-regression model when accounting for
         observed confounders and 6 substitute confounders estimated by PPCA.
      3. `coef_adni_original.csv`: Estimated coefficients of Beta-regression model when ignoring confounding.
      4. `coef_adni_age_residualized.csv`: Estimated coefficients of Beta-regression model when accounting for
         observed confounders via the regress-out approach.
      5. `coef_adni_combat_residualized.csv`: Estimated coefficients of Beta-regression model when harmonizing
         volume and thickness measures via the ComBat approach.


### Semi-Synthetic Simulation Study

1. Make sure your created `data/ukb-data.csv` as outlined above.
2. To execute all steps of the simulation study, you will need at least 64GB of RAM.
   Running the entire pipeline can take days and can be started by executing:
```bash
docker run -it --rm \
-v $(pwd)/data:/workspace/data \
-v $(pwd)/outputs:/workspace/outputs \
ghcr.io/ai-med/causal-effects-in-alzheimers-continuum:v0.2.0 \
./ukb-experiments.sh
```

3. The main result of the experiments will stored in the `outputs/ukb/results` folder.

   1. `ukb_visualize_output.ipynb`: The notebook contains a table of the Bayesian p-values for each model and latent dimension.
   Moreover, it will contain a table summarizing the bias in the estimates of the causal effects compared to the true causal effects.

   2. `experiments_summary.csv`: Table summarizing the bias in the estimates of the causal effects compared to the true causal effects.

   3. `all_experiments.h5`: Contains the bias of estimated causal effects for each individual experiment, i.e. ratio of direct to confounding effect,
   model, and repetition. To load the results for the direct to confounding
   effect ratio 10/1, use pandas:
   ```python
   results = pd.read_hdf("all_experiments.h5", key="x10_z1")
   ```
   The rows are the coefficients, and the columns are organized hierachically such that the first level is the experiment, the second level the model,
   and the third level the repitition.
