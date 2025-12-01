# Anomaly-Detection-in-Time-Series-Data-AWS-EC2-CPU-Utilization-



This project implements a **hybrid machine-learning anomaly detection system** for 10,000-point EC2 CPU utilization time-series datasets. I served as the **lead programmer**, responsible for the full Python implementation, model design, optimization, evaluation, and visualisation.

---

## 🔍 Project Overview
The objective was to accurately detect anomalous segments in multiple time-series files while ensuring strong generalization to unseen datasets.  
Our approach integrates:

- **Sliding-window feature engineering**
- **Ensemble ML models**  
  - Isolation Forest  
  - One-Class SVM  
  - Local Outlier Factor  
  - Elliptic Envelope
- **Statistical thresholding**
- **Range-based anomaly extraction**

Example output:

Anomaly 1: 2001 – 2005

Anomaly 2: 2010 – 2014


---

## 🚀 Key Features
- Hybrid ML + statistical anomaly detection pipeline  
- Numba-optimized sliding-window generation  
- Ensemble score aggregation  
- Automatic anomaly-range extraction  
- Visualisation of predicted vs. true anomalies  
- Full evaluation suite (F1, Precision, Recall)

---

## 📈 Model Performance
**Results across 10 test datasets:**

| Metric | Score |
|--------|--------|
| **Recall** | **94.2%** |
| **Precision** | **86.9%** |
| **F1-Score** | **90.1%** |

The system shows excellent sensitivity while maintaining strong precision.

---

## 🧠 My Role (Lead Programmer)
- Built and optimized the entire anomaly-detection pipeline  
- Developed sliding-window logic + normalization + ensemble scoring  
- Designed the hybrid model combining ML algorithms with statistical rules  
- Implemented full evaluation metrics and visualisation tools  
- Collaborated with teammates who handled presentation design, documentation, and annotations

---

## 🛠️ Tech Stack
- Python  
- Pandas, NumPy, Numba  
- scikit-learn  
- Matplotlib  
- Google Colab  

---






