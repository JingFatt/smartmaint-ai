# SmartMaint AI

Compressor anomaly detection on real metro-train data, served as a containerised API with model tracking and a model registry.

This is a portfolio project. It covers the path from raw sensor data to a deployed model service: data audit, modelling, evaluation, experiment tracking, a registered model, a FastAPI service with tests, and a Docker image. The evaluation is deliberately honest about what the data can and cannot show (see [Limitations](#limitations)).

## What it does

- Reads sensor readings from an air-production unit (compressor) of a metro train.
- Flags abnormal operation with two detectors that are compared side by side:
  - **Isolation Forest** (unsupervised, trained on clean normal data only)
  - **Rule baseline**: share of time the compressor was under load in the last 30 minutes
- Serves both through an HTTP API, loading the model version that the MLflow registry marks as `champion`.

## Architecture

```
MetroPT-3 CSV -> label + data-quality flags -> train / evaluate
                                                    |
                                       MLflow (runs, metrics, registry)
                                                    |  alias: champion
                                       export_champion.py (checks scores match)
                                                    |
                                    serving/model/{model.joblib, model_info.json}
                                                    |
                                  FastAPI service (Docker image, non-root, healthcheck)
                                   /health  /model-info  /predict
```

## Data

MetroPT-3 (UCI Machine Learning Repository, CC BY 4.0). Air compressor of a metro train, 15 signals, Feb to Sep 2020. The CSV used here is sampled every 10 seconds (1,516,948 rows); the original data was logged at 1 Hz. The data is unlabeled; the company's failure reports list 4 air-leak failures.

Citation:
- Veloso, B., Ribeiro, R.P., Pereira, P.M., Gama, J.: The MetroPT dataset for predictive maintenance. Scientific Data 9, 764 (2022). DOI 10.1038/s41597-022-01877-3
- Davari, N., Veloso, B., Ribeiro, R.P., Pereira, P.M., Gama, J.: Predictive maintenance based on anomaly detection using deep learning for air production unit in the railway industry. IEEE DSAA 2021. DOI 10.1109/DSAA53316.2021.9564181

### Data audit

Before modelling, the data was checked against the failure reports. Findings:

- **Frozen sensor stretches:** 7 stretches where all four analogue signals tested (TP2, TP3, Motor_current, Oil_temperature) stay exactly constant for 1 hour or more, about 140 hours in total (50,425 rows). These are data problems, not failures. One of them covers about 51 hours in late June.
- **Unreported sustained-load episodes:** 10 episodes of 1 hour or more with the compressor continuously under load that are not in the failure reports, about 86 hours in total. Several look like the reported failures. The data cannot tell whether they are real faults, so they are marked `uncertain` and excluded from both training and scoring.
- **Report vs data:** start and end times of failures 2 to 4 match the data within about 15 minutes. For failure 1, sustained load began about 24 minutes after the reported start and continued about 2 hours after the reported end.
- **Normal data is not stationary:** February looks different from later months (lower load, cooler oil), so training on clean normal data from all months matters.
- Result: 1,397,538 rows (about 92%) are usable as clean normal data.

## Method

- Split: clean normal data is split chronologically 70 / 15 / 15 into train, validation and normal test. The test set is the normal hold-out plus all reported failure rows that are not frozen or uncertain (29,863 failure rows).
- Isolation Forest on standardised features. The alarm threshold is the 99th percentile of scores on validation normal data (0.6765).
- Rule baseline: mean of `DV_eletric` over the last 180 readings (30 minutes) above 0.9.
- Event-style alarms use the same 180-row window: Isolation Forest alarms when more than half of the rows in the window are flagged.

## Results

Same test set for both models. Logged in MLflow experiment `smartmaint-anomaly`.

| Metric | Isolation Forest | Rule baseline |
|---|---|---|
| ROC-AUC | 0.998 | 0.999 |
| PR-AUC | 0.975 | 0.998 |
| Precision | 0.962 | 0.998 |
| Recall | 0.967 | 0.984 |
| False positives (rows) | 1127 | 71 |
| False negatives (rows) | 996 | 486 |
| Failure 1: rows flagged | 100% | 98.1% |
| Failure 2: rows flagged | 98.6% | 97.0% |
| Failure 3: rows flagged | 100% | 99.5% |
| Failure 4: rows flagged | 40.6% | 90.1% |

Precision and recall for the rule are computed from the logged false-positive and false-negative counts.

What the results say:

- **A one-line rule is as good as the Isolation Forest on this data**, and better on failure 4. The rule's two parameters were chosen after looking at the failure signature, so its numbers are optimistic. The Isolation Forest threshold used only normal validation data.
- The Isolation Forest's misses are almost all in failure 4, whose sensor profile differs from failures 1 to 3 (DV_pressure near 0, oil temperature 80 to 86 degrees C).
- **Alarms come after onset, not before.** Measured from the start of sustained load, alarms arrive roughly 15 to 30 minutes later for both methods, except the Isolation Forest on failure 4 (about 3 hours). This is fast detection, not early warning. Part of the delay comes from the 30-minute window.
- In the 31-day normal test period (August) there were 0 alarms for the Isolation Forest and 1 for the rule.

## Limitations

- Only 4 reported failures, and failure 4 differs from the others. Row-level metrics look large but there are only 4 independent events. The numbers do not extrapolate.
- The normal test period is August only, and suspicious stretches were removed from it, so the false-alarm rate is optimistic.
- The "uncertain" episodes may be real faults. They are excluded rather than judged.
- No ground truth for early warning: the reports give failure windows, not degradation onsets.
- Single compressor, single dataset.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service status and champion version |
| `GET /model-info` | Model version, threshold, alarm definitions, known limitations |
| `POST /predict` | Recent readings (oldest to newest, 10 s apart) in, scores and alarms out |

`/predict` needs 180 readings for a full 30-minute window. With fewer, scores are returned but both alarm fields are `null`. Input is validated (all 15 fields required, unknown fields and non-finite numbers rejected).

Interactive docs: `http://127.0.0.1:8000/docs`

<!-- Add screenshots: docs/figures/api_docs.png, docs/figures/mlflow_compare.png -->

## Run it

Environment (Python 3.11; versions used: scikit-learn 1.9.1, MLflow 3.18.0, FastAPI 0.162.7):

```
conda create -n SDUG python=3.11 -y
conda activate SDUG
pip install pandas numpy matplotlib scikit-learn jupyter pyarrow mlflow fastapi "uvicorn[standard]" pydantic httpx pytest
```

Data: download `MetroPT3(AirCompressor).csv` from the UCI page for MetroPT-3 into `data/raw/`.

Pipeline:

```
python models/anomaly_detection/prepare_data.py            # label failures
python models/anomaly_detection/flag_quality.py            # frozen / uncertain flags
python models/anomaly_detection/prepare_dataset.py         # train / validation / test split
python models/anomaly_detection/train_isolation_forest.py  # train + save artifacts
python models/anomaly_detection/evaluate_events.py > docs/eval_v1.txt
python mlops/log_models.py                                  # log runs, register model, set champion alias
python mlops/export_champion.py                             # export champion for serving
```

Inspect experiments:

```
mlflow ui --backend-store-uri sqlite:///mlops/mlflow.db
```

Test and serve:

```
python -m pytest -q
uvicorn backend.main:app --reload
python backend/demo_client.py
```

Docker (run `export_champion.py` first; the model file is not committed):

```
docker build -f backend/Dockerfile -t smartmaint-api:0.1.0 .
docker run -d --name smartmaint -p 8001:8000 smartmaint-api:0.1.0
python backend/demo_client.py http://127.0.0.1:8001/predict
```

The container returns the same results as the local service on the demo windows.

## Repository layout

```
backend/        FastAPI service, Dockerfile, pinned requirements, demo client
models/         data preparation, training and evaluation scripts
mlops/          MLflow logging and champion export
serving/model/  exported champion (model_info.json is committed, model.joblib is not)
tests/          API tests
docs/           eval output, figures, MetroPT-3 data description
```

## Model promotion gate

A new model version is registered as a challenger and only replaces `champion` if it passes a fixed gate (`mlops/promote.py`). The thresholds were fixed before the first challenger was trained:

1. No reported failure's detection rate drops by more than 2 percentage points.
2. False positives are at most 1.10x the champion's.
3. PR-AUC improves by at least 0.005, or the worst failure's detection rate improves by at least 5 points.

If a challenger is promoted, the old champion keeps the alias `previous` for rollback. Every decision is appended to `docs/promotion_log.csv`.

The first challenger (same model family, 300 trees, 4096 samples per tree) was rejected:

| Metric | Champion (v1) | Challenger |
|---|---|---|
| ROC-AUC | 0.998 | 0.986 |
| PR-AUC | 0.975 | 0.852 |
| Precision | 0.962 | 0.661 |
| Recall | 0.967 | 0.073 |
| False positives (rows) | 1127 | 1122 |
| Failures 1 / 2 / 3 / 4, rows flagged | 100 / 98.6 / 100 / 40.6 % | 0.6 / 0.5 / 9.2 / 32.7 % |

The challenger had almost the same false-positive count and an ROC-AUC only 0.013 lower, so a gate based on those two numbers would have let it through. The per-failure check caught it. The cause was not investigated; the ranking quality (ROC-AUC) is still decent, so the validation-based alarm threshold may not suit this configuration.

The gate is evaluated on the same four-failure test set as the main evaluation, so it demonstrates the mechanism. A production gate would need fresh data.

## Not done yet

- Drift monitoring and a scheduled retraining job
- CI pipeline (lint, tests, image build)
- Dashboard
- Windowed features for the Isolation Forest, to address failure 4
