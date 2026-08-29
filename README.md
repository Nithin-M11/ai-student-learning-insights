# AI-Powered Student Report & Learning Insights

An AI-powered student performance analysis system that predicts future academic performance and generates personalized learning insights, recommendations, and action plans.

## 🚀 Live Demo

[Open the Live Application](https://ai-student-learning-insights-ftdub3sdbhr59hepn9npdq.streamlit.app/)

## 📌 Project Overview

This project uses Machine Learning to analyze student academic and study-related data and provide meaningful insights about their performance.

The application allows users to select a student and view:

- Current academic performance
- AI-predicted future score
- Predicted performance level
- Subject-wise performance
- Current vs predicted score comparison
- Strongest and weakest subjects
- Study profile
- Personalized learning insights
- Learning recommendations
- Recommended action plan
- Downloadable HTML and PDF reports

## 🤖 Machine Learning

A **Random Forest Regressor** is used to predict the student's future performance score.

### Input Features

- Study hours
- Attendance
- Assignment score
- Previous score
- Mathematics
- DBMS
- Python
- Operating Systems
- Machine Learning

### Model Evaluation

The trained model achieved:

- **MAE:** 3.83
- **RMSE:** 4.82
- **R² Score:** 0.65

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- ReportLab

## 📊 Application Workflow

```text
Student Data
     ↓
Data Analysis
     ↓
Machine Learning Model
     ↓
Future Score Prediction
     ↓
Performance Analysis
     ↓
Personalized Learning Insights
     ↓
Recommendations & Action Plan
     ↓
Student Report
