# 📡 EMI Risk Analyzer

## AI-Based PCB EMI Prediction & Design Recommendation Platform

🔗 **Live Demo:**  
https://emiriskanalyzer-whxmg9ijhqjdcnmi5b4mua.streamlit.app/

---

## 📌 Overview

EMI Risk Analyzer is an AI-powered PCB analysis platform that predicts electromagnetic interference (EMI) risk levels based on PCB design parameters.

The system combines Machine Learning and Explainable AI (SHAP) techniques to analyze PCB characteristics, identify critical risk factors, and generate automated EMI analysis reports.

This project bridges **PCB design engineering** and **Data Science** by providing an intelligent approach to EMI risk evaluation.

---

## 🚀 Features

### 🤖 AI-Based EMI Risk Prediction
- Predicts PCB EMI risk as **PASS / FAIL**
- Provides confidence score and risk evaluation

### 📊 Interactive Dashboard
- Built with Streamlit
- Real-time PCB parameter analysis
- User-friendly visualization

### 🧠 Explainable AI with SHAP
- Identifies the most influential EMI risk factors
- Provides transparent model explanations

### 📄 Automated PDF Reporting
- Generates downloadable EMI analysis reports
- Includes prediction results and risk information

---

## ⚙️ Analyzed PCB Parameters

The system evaluates:

- Clock Frequency
- Rise Time
- Signal-GND Distance
- Trace Length
- Stub Length
- Differential Pair Symmetry
- Ground Plane
- Shielding
- Switching Regulator
- Layer Count

---

## 🧠 Machine Learning Workflow
PCB Design Parameters
|
↓
Data Processing
|
↓
Machine Learning Model
|
↓
EMI Risk Prediction
|
↓
SHAP Explainability
|
↓
PDF Report Generation


---

## 🛠️ Technologies Used

| Category | Technology |
|---|---|
| Programming Language | Python |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Explainable AI | SHAP |
| Visualization | Matplotlib |
| Reporting | FPDF |

---

## 📂 Project Structure

EMI_Risk_Analyzer/

│
├── app/
│ └── app.py # Streamlit application
│
├── src/
│ ├── predict.py # Prediction pipeline
│ ├── train_model.py # Model training
│ ├── explain_model.py # SHAP analysis
│ └── pdf_report.py # PDF generation
│
├── models/
│ └── emi_model.pkl # Trained ML model
│
├── data/
│ └── emi_dataset.csv # Dataset
│
├── notebooks/
│ └── emi_analysis.ipynb
│
└── requirements.txt


---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/sevvalbir/EMI_Risk_Analyzer.git 

Install dependencies:

```bash
pip install -r requirements.txt

Run the application:
streamlit run app/app.py
