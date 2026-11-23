# Predicting Early Hospital Readmission of Diabetic Patients

Machine learning project, ESILV A4 (2026–2027). We predict whether a hospital encounter of a diabetic patient will be followed by a **readmission within 30 days**, using only the information available at discharge.

**Team:** Thierno Diallo · Noe Payan · Joe Vaseekaran · Lisandru Zuccarelli

## Contents

- [Problem](#problem)
- [Dataset](#dataset)
- [Method](#method)
- [Results so far](#results-so-far)
- [Getting started](#getting-started)
- [Repository structure](#repository-structure)
- [Roadmap](#roadmap)
- [Limitations](#limitations)
- [References](#references)

## Problem

Early readmissions are costly and can signal that a patient needed closer follow-up. A hospital cannot contact every patient after discharge, so a risk score could help the follow-up team decide whom to review first.

**Main question:** can a model identify the encounters followed by a readmission within 30 days, and do better than a simple logistic regression?

**Second question:** which variables are the most useful, and does a smaller feature set give similar results?

| | |
| --- | --- |
| Task | Supervised binary classification |
| Positive class (`y = 1`) | `readmitted == "<30"` (11.4 % of training encounters) |
| Negative class (`y = 0`) | `readmitted` is `">30"` or `"NO"` |
| Prediction point | Hospital discharge |
| Main metric | PR-AUC (average precision) |

## Dataset

[Diabetes 130-US Hospitals for Years 1999–2008](https://doi.org/10.24432/C5230J) (UCI Machine Learning Repository): 101,766 encounters of 71,518 patients, 50 columns covering demographics, admission details, previous visits, diagnoses, lab results and diabetes medications.

The data is included in the repository, so there is nothing to download:

| File | Content |
| --- | --- |
| [diabetic_data.csv](diabetic_data.csv) | Raw encounters |
| [IDS_mapping.csv](IDS_mapping.csv) | Meaning of the admission and discharge codes |
| [diabetic_clean.csv.gz](Week_2/outputs/diabetic_clean.csv.gz) | Clean dataset, with the fixed `split` and `fold` assignment |
| [feature_config.json](Week_2/outputs/feature_config.json) | Feature lists, dropped columns and split parameters |

## Method

1. **Cohort.** Encounters ending in death or hospice are removed, because readmission cannot be observed normally for them. This leaves 99,343 encounters of 69,990 patients.
2. **Row-wise cleaning.** `?` becomes missing, age intervals become midpoints, ICD-9 codes are grouped into 9 clinical categories (plus `Missing`), and tests that were not taken become `NotMeasured`.
3. **Split by patient.** 20 % of the patients are held out as a test set (`GroupShuffleSplit`). The others form 5 cross-validation folds (`StratifiedGroupKFold`). The seed is fixed to 42.
4. **Features.** 43 features are kept (9 numerical, 34 categorical). `weight` (97 % missing) and three medications that are constant in the training data are dropped.
5. **Preprocessing pipeline.** Median imputation and standardisation for numerical features; an `Unknown` level and one-hot encoding for categorical features, with rare levels merged.
6. **Baselines.** A majority-class classifier and a logistic regression with `class_weight="balanced"`.

### Preventing data leakage

| Risk | Safeguard |
| --- | --- |
| Same patient in training and evaluation data | Every split is grouped by `patient_nbr` and checked with assertions |
| Statistics learned from evaluation data | Imputation, scaling and encoding are refitted inside each training fold |
| Test set influencing decisions | Feature decisions use training patients only; the test set is reserved for one final evaluation in week 4 |
| Identifiers or target used as features | `encounter_id`, `patient_nbr` and `readmitted` are excluded |
| Undetected leak | With shuffled labels, the pipeline falls back to chance (PR-AUC 0.115, ROC-AUC 0.496) |

## Results so far

5-fold cross-validation on the training patients (79,541 encounters), mean ± standard deviation over folds. The test set has not been evaluated yet.

| Metric | Majority class | Logistic regression |
| --- | --- | --- |
| **PR-AUC** (main metric) | 0.114 ± 0.000 | **0.214 ± 0.002** |
| ROC-AUC | 0.500 ± 0.000 | **0.663 ± 0.007** |
| Precision, top 20 % flagged | 0.117 ± 0.006 | **0.215 ± 0.004** |
| Recall, top 20 % flagged | 0.204 ± 0.010 | **0.376 ± 0.007** |
| Balanced accuracy (threshold 0.5) | 0.500 ± 0.000 | **0.616 ± 0.008** |
| Recall (threshold 0.5) | 0.000 ± 0.000 | **0.548 ± 0.015** |
| Accuracy (threshold 0.5) | **0.886 ± 0.000** | 0.668 ± 0.004 |

- **Accuracy is misleading.** The majority class has the best accuracy but never finds an early readmission.
- **The logistic regression almost doubles the PR-AUC.** Flagging the 20 % highest-risk encounters catches about 38 % of early readmissions, and about 1 flagged encounter in 5 is a true one (about 1 in 9 at random).
- **No sign of over-fitting.** Training PR-AUC is 0.221 against 0.214 in validation, which motivates more flexible models in week 3.
- **Previous inpatient visits** have the largest coefficient among the numerical features. This is an association, not a cause.

All metrics are in [baseline_cv_results.csv](Week_2/outputs/baseline_cv_results.csv).

## Getting started

Requires **Python 3.13+**. We recommend [uv](https://docs.astral.sh/uv/getting-started/installation/), which installs the exact versions from `uv.lock`.

```bash
git clone https://github.com/diallothierno0223/ml_project_esilv.git
cd ml_project_esilv
```

### With uv (recommended)

```bash
uv sync
uv run --with jupyterlab jupyter lab
```

In VS Code, run `uv sync` and select the `.venv` interpreter as the notebook kernel.

### With pip

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install . jupyterlab
jupyter lab
```

### With conda

```bash
conda create -n ml-project-esilv python=3.13
conda activate ml-project-esilv
pip install . jupyterlab
jupyter lab
```

### Reproduce the results

Open [week2_data_preparation_baseline.ipynb](Week_2/week2_data_preparation_baseline.ipynb) and run all cells. This overwrites the files in `Week_2/outputs/`.

### Reuse the clean dataset

Use the saved `split` and `fold` columns so that every model is compared on the same data. From the repository root:

```python
import json
import pandas as pd

with open("Week_2/outputs/feature_config.json") as f:
    config = json.load(f)
categorical = config["categorical"]
features = config["numeric"] + categorical

data = pd.read_csv(
    "Week_2/outputs/diabetic_clean.csv.gz",
    dtype={c: "object" for c in categorical},  # admission codes are categories, not numbers
)

train = data[data["split"] == "train"]         # keep the test set for week 4
X, y, folds = train[features], train["target"], train["fold"]
```

## Repository structure

```text
ml_project_esilv/
├── Week_1/
│   ├── project_specification.pdf             # Problem, objectives, evaluation plan
│   └── prompts.md                            # AI prompts (in French)
├── Week_2/
│   ├── week2_data_preparation_baseline.ipynb # Cleaning, split, baselines, results
│   └── outputs/
│       ├── diabetic_clean.csv.gz
│       ├── feature_config.json
│       └── baseline_cv_results.csv
├── src/ml_project_esilv/                     # Python package (placeholder)
├── diabetic_data.csv                         # Raw data
├── IDS_mapping.csv                           # Code descriptions
├── Project_Scope_Early_Readmission_Risk.docx # Business case and scope
├── Project-Guidelines-ML-2026.pdf            # Course guidelines
├── project_timeline.pdf                      # Milestones and grading
├── pyproject.toml                            # Dependencies
└── uv.lock                                   # Locked versions
```

## Roadmap

- [x] **Week 1: definition.** Topic, dataset, project specification
- [x] **Week 2: data and baseline.** Cleaning, patient-level split, baselines
- [ ] **Week 3: models.** Random forest, gradient boosting, L1 logistic regression, tuning, and filter / wrapper / embedded feature selection
- [ ] **Week 4: final evaluation.** Single test-set evaluation, fairness by age, sex and race, report and presentation

## Limitations

- The data is old (1999–2008) and US-only, so results may not generalise to current patients or other hospitals.
- `class_weight="balanced"` changes the probability scale: baseline scores rank patients but are not calibrated probabilities.
- Some coefficients come from rare categories and should be read carefully.
- The analysis is retrospective and shows associations, not causes.

This is an academic proof of concept, **not a medical device**. It must not be used for clinical decisions.

## References

1. J. Clore, K. Cios, J. DeShazo and B. Strack (2014). *Diabetes 130-US Hospitals for Years 1999–2008* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5230J>
2. B. Strack et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records*. BioMed Research International, 781670. <https://doi.org/10.1155/2014/781670>
