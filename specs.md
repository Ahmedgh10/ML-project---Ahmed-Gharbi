# Machine Learning: Anomaly detection in sensor data

**Author:** Ahmed Gharbi

---

## Context and Objectives of the Application

### Context
Smart systems continuously generate operational and network data such as CPU usage, memory consumption, traffic rates, and protocol indicators. Under normal conditions, these metrics follow stable patterns. During attacks, however, noticeable changes occur—for example, traffic spikes during DoS attacks or abnormal values during injection attacks. Due to the large volume and diversity of data produced by smart devices, manual monitoring is inefficient. Therefore, an automated ML-based detection solution is needed to ensure fast, accurate, and scalable identification of abnormal or malicious behavior in smart environments.

### Objectives
The main objective of this mini-project is to develop an intelligent Python application that detects and classifies anomalous activity in smart-system data using Machine Learning techniques. The application aims to:

* **1. Prepare and analyze a labeled dataset** relevant to smart-system anomaly detection:
  * explore the dataset structure, feature types, and class distribution,
  * clean and transform the data to make it suitable for ML training.
* **2. Build and evaluate multiple predictive models** from different algorithmic families:
  * train at least three supervised learning models (e.g., linear model, tree ensemble, boosting model),
  * tune key hyperparameters and compare performance using appropriate evaluation metrics.
* **3. Select a final model** that provides the best trade-off between:
  * detection performance (e.g., F1-score, precision/recall),
  * robustness to class imbalance,
  * reasonable model complexity and usability in the project context.
* **4. Implement a usable application prototype** (Python-based) that can:
  * load and preprocess input data consistently,
  * output predictions (normal vs attack class),
  * provide evaluation outputs and/or user-friendly results presentation for demonstration.

> Overall, the project demonstrates a complete Machine Learning workflow—from dataset preparation to model training and evaluation—applied to a realistic cybersecurity problem in smart environments, with the goal of supporting early detection of threats and improving system resilience.

---

## Presentation of the Dataset Used and Its Characteristics

### Dataset source and justification
For this project, we used the "Anomaly Detection and Threat Intelligence Dataset (SmartSysCTI)" available on Kaggle: https://www.kaggle.com/datasets/ziya07/anomaly-detection-and-threat-intelligence-dataset This dataset is appropriate for our objective because it simulates realistic activity logs and operational/network behavior of smart/IoT systems. It contains labeled examples of normal behavior and cyberattacks (anomalies), which makes it suitable for training supervised Machine Learning models for anomaly and intrusion detection.

### Dataset structure
* **File used:** `smart_system_anomaly_dataset.csv`
* **Number of instances (rows):** to be filled from your notebook/script output (`df.shape[0]`)
* **Number of attributes (columns):** 14 columns (including the target)

### Attributes (features)
The dataset includes the following variables:

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `timestamp` | temporal | event time |
| `device_id` | categorical/identifier | device identifier |
| `device_type` | categorical | device category (sensor, camera, etc.) |
| `cpu_usage` | numerical | CPU usage level |
| `memory_usage` | numerical | memory usage level |
| `network_in_kb` | numerical | inbound network traffic (KB) |
| `network_out_kb` | numerical | outbound network traffic (KB) |
| `packet_rate` | numerical | packets per time unit |
| `avg_response_time_ms` | numerical | average response time (ms) |
| `service_access_count` | numerical | number of service access requests |
| `failed_auth_attempts` | numerical | failed authentication attempts count |
| `is_encrypted` | binary/categorical | indicates whether traffic is encrypted |
| `geo_location_variation` | numerical | variation in geographic location patterns |
| `label` | target | class label (normal behavior or attack type) |

### Target variable
* **Target column:** `label`
* **Type of task:** supervised classification (binary or multiclass depending on the label values)

### Data distribution
To analyze balance/imbalance in the dataset, the distribution of the target classes is computed using:
* `value_counts()` to obtain exact counts and percentages
* a bar chart (`seaborn.countplot`) to visualize how frequent each class is

In the final report, the class distribution plot should be included, and the presence of any imbalance should be discussed because it can strongly influence the evaluation metrics and model performance.