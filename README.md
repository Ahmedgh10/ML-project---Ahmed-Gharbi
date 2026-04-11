# Smart System Anomaly Detection (IoT)

End-to-end mini-project for smart-system / IoT anomaly detection:

- Train and compare multiple supervised models (multi-class).
- Evaluate with realistic split regimes (random vs time vs unseen-device).
- Serve predictions via a Streamlit web app.
- Provide simple, actionable explanations for anomalies using a “nearest normal twin” + feature deltas.

## What’s inside

- Training CLI: `train.py`
- Streamlit app: `app.py`
- Core ML code: `ml/` (data prep, modeling, inference)
- Deployment artifacts: `artifacts/` (committed for Streamlit Cloud)

## Labels (task)

The model predicts one of:

- `Normal`
- `Anomaly_DoS`
- `Anomaly_Injection`
- `Anomaly_Spoofing`

## Install

```bash
python -m pip install -r requirements.txt
```

## Train (and write artifacts)

```bash
python train.py --data smart_system_anomaly_dataset.csv --artifact-dir artifacts --regime random
```

Split regimes:

- `random`: stratified split (reference)
- `time`: forward-in-time split (more realistic)
- `group`: unseen-device split (split by `device_id`)

Output artifacts (saved into `artifacts/`):

- `model.joblib`: trained pipeline (preprocessing + classifier)
- `normal_vectors.npy`: transformed vectors for normal samples
- `normal_explain.csv`: metadata + feature values for normal samples (used for explanations)
- `train_summary.json`: training summary (best model, scores)

## Run the Streamlit app (local)

```bash
streamlit run app.py
```

In the UI:

- Upload a CSV with the same columns as the dataset.
- The `label` column is optional; if provided, the app shows an evaluation report.

## Deploy on Streamlit Community Cloud

This repo is deployable as-is.

- Repo: `Ahmedgh10/ML-project---Ahmed-Gharbi`
- Branch: `main`
- Main file: `app.py`

Notes:

- `runtime.txt` pins Python 3.11 to avoid scientific stack issues.
- The raw dataset is not committed; the app works on any user-uploaded CSV.

## Repository layout

```text
.
├─ app.py
├─ train.py
├─ requirements.txt
├─ runtime.txt
├─ artifacts/
└─ ml/
	├─ data.py
	├─ modeling.py
	└─ inference.py
```
