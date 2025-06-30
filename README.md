# BioTrack: Heart Attack Risk Prediction System

**BioTrack** is a machine learning-based application built to predict heart attack risk using clinical and lifestyle health data. With a focus on **accuracy**, **explainability**, and **user accessibility**, BioTrack empowers individuals and healthcare professionals with real-time predictions and insight into contributing health factors.

---

## 🚀 Features

- **High-Accuracy Prediction** — Achieves **94% accuracy** on the test set  
- **SHAP Explainability** — Understand **why** a prediction was made  
- **Hybrid Model** — Combines **Random Forest** + **Deep Neural Network (DNN)**  
- **Evaluation Tools** — Confusion matrix, ROC AUC, F1 score, and more  
- **Web-Based Interface** — (Planned) Interactive UI via **Streamlit** or **Flask**  
- **Secure & Scalable** — Architecture ready for deployment and scaling

---

## Tech Stack

- **Languages**: Python  
- **Model**: Hybrid of `RandomForestClassifier` & custom `DNN`  
- **Libraries**:  
  - `scikit-learn`  
  - `pandas`, `numpy`  
  - `shap`, `matplotlib`, `seaborn`  
  - `torch` (for DNN)

---

## Model Performance

- **Accuracy**: `94%` on held-out test data  
- **Metrics**:
  - Precision, Recall, F1 Score, AUC-ROC  
  - Confusion Matrix  
  - Predicted vs Actual Probability plots

- **Explainability**:
  - SHAP bar plots & beeswarm plots for **global** and **individual** feature impact  
  - Plots auto-saved as `.png` for easy visualization
 
---

 ## Dataset Attribution

This project uses the [Patients Data for Medical Field](https://www.kaggle.com/datasets/tarekmuhammed/patients-data-for-medical-field) dataset by **Tarek Muhammed** (Kaggle).  
The dataset is used under Kaggle’s [terms of use](https://www.kaggle.com/terms) for research and educational purposes.

---
