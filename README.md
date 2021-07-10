# Estimation of Causal Effects in the Presence of Unobserved Confounding in the Alzheimer's Continuum

[![Preprint](https://img.shields.io/badge/arXiv-2006.13135-b31b1b)](https://arxiv.org/abs/2006.13135)
[![Paper](https://img.shields.io/static/v1?label=DOI&message=10.1007%2f978-3-030-78191-0_4&color=3a7ebb)](https://dx.doi.org/10.1007/978-3-030-78191-0_4)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

This repository contains the code to the paper "Estimation of Causal Effects in the Presence of Unobserved Confounding in the Alzheimer's Continuum."
If you are using this code, please cite:

```
@inproceedings(Poelsterl2021-causal-effects-in-ad,
  title     = {{Estimation of Causal Effects in the Presence of Unobserved Confounding in the Alzheimer's Continuum}},
  author    = {P{\"{o}}lsterl, Sebastian and Wachinger, Christian},
  booktitle = {Information Processing in Medical Imaging},
  year      = {2021},
  pages     = {45--57},
  doi       = {10.1007/978-3-030-78191-0_4},
  url       = {https://arxiv.org/abs/2006.13135},
}
```

## Installation

1. Install [Docker](https://docs.docker.com/install/).
2. Build Docker image `causalad`:
```bash
docker build -t causalad .
```

If you want to use the code for development, you can use [conda](https://conda.io/miniconda.html)
to create an environment with all dependencies from `requirements.yaml`.

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
2. Download `ADNIMERGE.CSV`, `APOERES.csv` and `UPENNBIOMK_MASTER.csv`.
3. Copy `APOERES.csv` to the `data/` directory.
4. Use `ABETA`, `PTAU`, `TAU` from `UPENNBIOMK_MASTER.csv` to determine
   which patients have an Alzheimer's pathologic by
  creating a column `ATN_status` that describes the A/T/N scheme, e.g.
  `A+/T+/N-` if `ABETA ≤ 192`, `PTAU ≥ 23`, and `TAU < 93`, following the
  thresholds from [Ekman et al., 2018](https://doi.org/10.1038/s41598-018-26151-8):
  > The individual CSF values were considered pathological (+)
  > if ≤192 pg/ml for Aβ42, ≥93 pg/ml for t-tau, and ≥23 pg/ml for p-tau.

5. Download T1 structural brain MRI from the [ADNI Data Portal](http://adni.loni.usc.edu/)
   and segment each with [FreeSurfer 5.3](https://www.freesurfer.net/) to obtain volume
   and thickness measurements.
6. Fill in the values of `data/adni-data-template.csv` by taking `ABETA`, `PTAU`, `TAU`
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
2. The workflow is split into 3 notebooks that have to be executed
   sequentially. Begin by starting the Jupyter notebook server:
```bash
docker run --rm -p 8888:8888 -v $(pwd)/data:/notebooks/data causalad
```
3. Go to https://localhost:8888 or click on the link that is printed
   when running the above command.
4. Click on the `adni-estimate-substitute.ipynb` notebook, which will
   open a new tab. In the menu at the top, go to *Cell* and click *Run All*.
   Once completed, this will create `data/adni-transformed.csv` and 4 files
   in `data/outputs/adni/dim6/`:

      1. `adni_aug_BPMF.csv`: Transformed features with 6 substitute confounders estimated by BPMF.
      2. `adni_aug_PPCA.csv`: Transformed features with 6 substitute confounders estimated by PPCA.
      3. `adni_aug_regressout.csv`: Transformed features with observed confounders regressed out
         by linear regression.
      4. `pvalue.csv`: Bayesian p-value of posterior predictive check for BPMF and PPCA.

5. Go back to https://localhost:8888, and click on the `adni-estimate-effects.ipynb` notebook.
   In the new tab, go to *Cell* and click *Run All*. Once completed, 4 additional files are
   created in `data/outputs/adni/dim6/`:

      1. `coef_bpmf.csv`: Estimated coefficients of Beta-regression model when accounting for
         observed confounders and 6 substitute confounders estimated by BPMF.
      2. `coef_noconf.csv`: Estimated coefficients of Beta-regression model when ignoring confounding.
      3. `coef_ppca.csv`: Estimated coefficients of Beta-regression model when accounting for
         observed confounders and 6 substitute confounders estimated by PPCA.
      4. `coef_regout.csv`: Estimated coefficients of Beta-regression model when accounting for
         observed confounders via the regress-out approach.

6. Finally, go back to https://localhost:8888 and open the `adni-compare-effects.ipynb` notebook.
   In the new tab, go to *Cell* and click *Run All*. This will display a figure comparing the
   estimated credible intervals for each model from the previous step. In addition, it writes
   the figure to `data/outputs/adni/dim6/coef-horizontal.pdf`.


### Semi-Synthetic Simulation Study

1. Make sure your created `data/ukb-data.csv` as outlined above.
2. To execute all steps of the simulation study, you will need at least 64GB of RAM.
   The entire process takes approximately 4 hours and can be started by executing:
```bash
docker run --rm -v $(pwd)/data:/notebooks/data causalad ./run-synthetic-ukb.sh
```
3. This will create 4 files in `data/outputs/synthetic_ukb/`.

    1. `augmented_data_bpmf_dim5.pkl`: Transformed features with 5 substitute confounders estimated by BPMF.
    2. `augmented_data_ppca_dim5.pkl`: Transformed features with 5 substitute confounders estimated by PPCA.
    3. `coefs_dim5.pkl`: Estimated coefficients of all models for 1000 different simulated outcomes.
    4. `evaluation_coefs_dim5.csv`: RMSE of coefficients with respect to true coefficients for each model,
       across 1000 simulations.
