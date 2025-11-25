
**DEEP LEARNING (CT-468)**

**SOLAR POWER GENERATION USING TIME SERIES**


**Submitted By:**

**Fatima Azeemi (AI-22001)**

**Fizzah Masroor (AI-22020)**

**Dua Sharif (AI-22303)**

**Fatima Tasneem (AI-22304)**

 
**Under Supervision Of:**

**Dr. Murk Marvi**  
 

**Department of Computer Science and Information Technology**

**NED University of Engineering & Technology, Karachi**
***


# 🌞 Solar Power Generation Forecasting using Deep Learning & Meta-Learning

### *(Teacher–Student Hybrid Architecture with Drift Detection & Explainable AI)*

This project presents a robust deep learning framework for **hourly solar power generation forecasting** using a hybrid **teacher–student stacking architecture**. The system integrates **CNN, LSTM, and TCN** sequence models as student learners, whose predictions are combined and passed to an **XGBoost teacher model** for high-accuracy meta-prediction. The approach significantly improves performance over traditional models and standalone deep learning architectures.

Additionally, the project includes **explainability (SHAP)**, **data drift detection**, and an interactive **Streamlit web application** for real-time forecasting and monitoring—making it suitable for industrial and utility-scale deployment.

---

## 🔍 **Project Overview**

### ⭐ Motivation

Accurate solar forecasting is essential for:

* Grid stability and load balancing
* Energy market trading
* Renewable energy integration
* Reducing carbon emissions and energy waste

Solar data is **highly nonlinear, seasonal, and weather-dependent**, making deep learning a strong candidate for improved forecasting accuracy.

---

## 📊 **Key Features**

### 🧠 **1. Teacher–Student Meta-Learning Architecture**

* **Students:** CNN, LSTM, TCN sequence models
* **Teacher:** XGBoost meta-model
* Student predictions are used as **meta-features**
* Produces higher accuracy and lower variance than individual models

### ⏱ **2. Time-Series Forecasting**

* 168-hour (7-day) look-back window
* Weather + irradiance + temporal cyclic features

### 🧼 **3. Data Preprocessing & Feature Engineering**

* Merging meteorological and power datasets
* Cyclic encodings (hour_sin, day_cos, etc.)
* MinMax normalization & log1p target scaling
* Handling nighttime zero-generation and sensor noise

### 🧪 **4. Evaluation Metrics**

* MAE, RMSE
* Masked MAPE (>10 MW threshold)

### 🩻 **5. Explainable AI (XAI)**

* SHAP global and local interpretability
* Feature importance visualization
* Interpretation of student model reliability

### ⚠️ **6. Data Drift Detection**

* **MMD (Maximum Mean Discrepancy)** for feature drift
* **Kolmogorov–Smirnov test** for target drift
* Drift alerts for system re-training

### 🖥 **7. Streamlit Frontend**

* Upload CSV files for real-time forecasting
* Interactive plots and drift visualizations
* Real-time metrics and monthly summaries
* Ready for local or cloud deployment

---

## 📈 **Performance Summary**

| Model                             | MAE (MW)   | RMSE (MW)  | Masked MAPE (%) |
| --------------------------------- | ---------- | ---------- | --------------- |
| **XGBoost Teacher (Final Model)** | **250.63** | **466.67** | **25.59**       |

Compared to baseline deep learning models and results from prior literature, the proposed approach achieves **state-of-the-art performance**, outperforming many published works.

---

## 📚 **Technologies Used**

* Python, NumPy, Pandas
* TensorFlow (CNN, LSTM, TCN)
* XGBoost
* SHAP
* Scikit-learn
* SciPy
* Streamlit
* Plotly

---

## 🚀 **Deliverables**

* Full training + inference pipeline
* Saved models (Students + Teacher)
* Streamlit dashboard
* Data drift monitoring system
* SHAP explainability visualizations
* Comprehensive documentation and codebase


