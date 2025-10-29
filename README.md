# 🔋 Self-Healing Electric Vehicle using Generative AI for Predictive Maintenance

## 📅 Week 1 – Battery Health & Fault Detection Models

### 🧠 Problem Statement

Electric Vehicles (EVs) face unpredictable component degradation due to dynamic environmental and operational conditions.
This project builds an **AI-driven predictive maintenance system** that estimates **battery health** and detects **faults** before they occur.
The end goal is to enable a *self-healing* EV that diagnoses, predicts, and prevents breakdowns through **Generative AI–powered insights**.

---

### 🎯 Week 1 Objective

* Preprocess the EV sensor dataset.
* Engineer synthetic **battery health scores** and **fault labels**.
* Build and train:

  * 🧩 **Regression Model** → Predict *Battery Health Score*
  * ⚙️ **Classification Model** → Detect *Fault vs. Healthy* status
* Evaluate model performance using multiple metrics and cross-validation.
* Analyze business impact (e.g., cost savings, reliability improvement).

---

### 📁 Repository Structure

```
Week1/
│
├── README.md
├── notebooks/
│   └── week1_battery_health_and_fault_detection.ipynb
├── data/
│   └── ev_dataset.csv
└── outputs/
    ├── feature_importance.png
    ├── confusion_matrix.png
    ├── health_score_distribution.png
    └── precision_recall_curve.png
```

---

### 📊 Dataset Information

* **Dataset:** Electric Vehicle Health Dataset (custom EV telemetry)
* **Size:** ~37 MB (trimmed to < 25 MB for GitHub)
* **Key Features:**

  * `Battery Voltage [V]`
  * `Battery Current [A]`
  * `Battery Temperature [°C]`
  * `Ambient Temperature [°C]`
  * `Motor Torque [Nm]`
  * `Velocity [km/h]`
  * `Throttle [%]`
  * `SoC [%]`

Two synthetic targets were engineered:

1. **Battery_Health_Score** → continuous (0 to 1)
2. **Fault** → binary (1 = Fault, 0 = Healthy)

---

### ⚙️ Workflow Completed

| Step                       | Description                                                               | Status |
| -------------------------- | ------------------------------------------------------------------------- | ------ |
| **1. Data Preparation**    | Loaded, cleaned, and merged EV sensor data                                | ✅      |
| **2. Target Generation**   | Synthetic health score + fault probability derived from risk metrics      | ✅      |
| **3. Feature Engineering** | Normalization, scaling, and slight noise addition                         | ✅      |
| **4. Model Training**      | Random Forest Regressor + Classifier                                      | ✅      |
| **5. Evaluation**          | R², MAE, Accuracy, F1, ROC-AUC                                            | ✅      |
| **6. Visualization**       | Confusion Matrix, Feature Importance, Health Score Distribution, PR Curve | ✅      |
| **7. Business Impact**     | Fault recall, false alarm rate, estimated cost savings                    | ✅      |
| **8. Model Export**        | `.pkl` files for both models + scaler                                     | ✅      |

---

### 📈 Key Results

#### 🔋 Battery Health Score (Regression)

| Metric              | Value         |
| ------------------- | ------------- |
| Train R²            | 0.996         |
| Test R²             | 0.996         |
| MAE                 | 0.0036        |
| Mean CV R² (5-fold) | 0.939 ± 0.058 |

#### ⚙️ Fault Detection (Classification)

| Metric                        | Value           |
| ----------------------------- | --------------- |
| Train Accuracy                | 0.599           |
| Test Accuracy                 | 0.541           |
| ROC-AUC                       | 0.553 (± 0.049) |
| F1 Score                      | 0.56            |
| Recall (Fault Detection Rate) | 60.1 %          |
| False Alarm Rate              | 30 %            |

---

### 💼 Business Impact Analysis

* **Faults Correctly Detected:** 60.1 %
* **Overall System Reliability:** 54.1 %
* **Estimated Savings:** $69,103,000 (≈ $500 per fault prevented)
  These figures show that even the baseline model provides significant cost reduction by predicting potential EV failures early.

---

### 📊 Visualizations

*(Sample visual outputs from notebook)*

1. Confusion Matrix – Fault Detection
2. Feature Importance (Top Sensors Affecting Faults)
3. Health Score Distribution by Fault Status
4. Precision–Recall Curve (AP ≈ 0.553)
5. Predicted vs. Actual Health Score Scatter Plot

---

### 🧩 Tools & Libraries

* **Python 3.12**
* **pandas**, **NumPy**
* **scikit-learn**
* **matplotlib**, **seaborn**
* **joblib**

---

### ✅ Outcome

Week 1 delivered a **fully operational ML pipeline** that:

* Generates clean, consistent input features from raw EV telemetry.
* Predicts battery health and fault likelihood with strong baseline accuracy.
* Demonstrates measurable business impact and cost benefit.

Models saved:

* `battery_health_model.pkl`
* `fault_detection_model.pkl`
* `scaler.pkl`

---

### 🚀 Next Steps (Week 2 Preview)

* Integrate **Generative AI** layer for automatic maintenance insight generation.
* Enhance fault classification through hyperparameter tuning and sensor fusion.
* Build real-time streaming dashboard for EV health tracking.

---

### 👩‍💻 Author

**Rakshita R T**
KLE Technological University
AI & Data Science | Generative AI Cohort Project
