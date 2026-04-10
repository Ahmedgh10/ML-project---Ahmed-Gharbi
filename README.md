# ML-project---Ahmed-Gharbi

This repo trains a multiclass anomaly classifier (Normal / Anomaly_DoS / Anomaly_Injection / Anomaly_Spoofing) and serves predictions via a minimal Streamlit app, including a simple explanation: the closest “normal twin” and feature deltas.

## Setup

```bash
pip install -r requirements.txt
```

## Train + save artifacts

```bash
python train.py --data smart_system_anomaly_dataset.csv --artifact-dir artifacts --regime random
```

Regimes:
- `random`: stratified split
- `time`: forward-in-time split
- `group`: unseen-device split (split by `device_id`)

## Run the app

```bash
streamlit run app.py
```

Upload a CSV with the same columns as the dataset (the `label` column may be missing).
