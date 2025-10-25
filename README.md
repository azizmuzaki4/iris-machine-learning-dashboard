# 🌸 Iris Machine Learning Dashboard

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-blueviolet)](https://iris-machine-learning-dashboard.streamlit.app/)

An **interactive machine learning dashboard** built with **Streamlit** and **Plotly** to train, evaluate, and predict **Iris flower species** using multiple classification algorithms.

---

## 🖼️ Dashboard Preview

| | |
|:--:|:--:|
| ![Main Page](https://github.com/azizmuzaki4/iris-machine-learning-dashboard/blob/main/0_main_page.png) | ![Boxplot Visualization](https://github.com/azizmuzaki4/iris-machine-learning-dashboard/blob/main/1_visualization_page_boxplot_per_feature.png) |
| ![Histogram & Scatter Plot](https://github.com/azizmuzaki4/iris-machine-learning-dashboard/blob/main/1_visualization_page_histogram_scatter_plot.png) | ![Heatmap & Pairplot](https://github.com/azizmuzaki4/iris-machine-learning-dashboard/blob/main/1_visualization_page_pairplot_heatmap_feature_correlation.png) |
| ![Training Radar Chart](https://github.com/azizmuzaki4/iris-machine-learning-dashboard/blob/main/2_training_page_cross_validation_result_radar_chart.png) | ![Confusion Matrix](https://github.com/azizmuzaki4/iris-machine-learning-dashboard/blob/main/3_evaluation_page_performance_model_on_test_set_confusion_matrix.png) |
| ![Prediction Page](https://github.com/azizmuzaki4/iris-machine-learning-dashboard/blob/main/4_prediction_page_iris_species_prediction_with_explanation.png) | |

---

## 🚀 Overview

This project demonstrates a complete **end-to-end machine learning workflow** for the Iris dataset.  
It allows users to explore data, train models, evaluate performance, and make predictions — all through an elegant Streamlit interface.

The dashboard is perfect for **learning, experimentation, and model comparison**, offering clear visualizations for every ML step.

---

## 🧠 Key Features

### 📊 Interactive Data Exploration
- Histograms, scatter plots, box plots, and correlation heatmaps  
- Color-coded species comparison  

### ⚙️ Model Training with Cross-Validation
- Supports multiple algorithms:
  - Support Vector Machine (SVM)
  - Random Forest
  - Gradient Boosting
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Naive Bayes  
- Automated hyperparameter tuning via `GridSearchCV`
- Selectable scoring metrics: Accuracy, F1, Precision, Recall

### 📈 Model Evaluation Dashboard
- Performance metrics on test set  
- Confusion Matrix for each model  
- Multi-class ROC Curve visualization  
- Side-by-side model comparison with data bars  

### 🔮 Prediction Page
- Predict species from user-input flower measurements  
- Confidence visualization using dynamic gauge  
- Position input within feature space (scatter plot)  
- Narrative explanation comparing to class averages  

---

## 🧩 Tech Stack

| Category | Tools & Libraries |
|-----------|------------------|
| Frontend UI | Streamlit, HTML, CSS |
| Visualization | Plotly Express, Plotly Graph Objects |
| Machine Learning | scikit-learn (SVC, RandomForestClassifier, etc.) |
| Data Handling | pandas, numpy |
| Optimization | GridSearchCV, Pipeline |
| Styling | Custom CSS, Light/Dark mode |

---

## 📁 Dataset

The dashboard uses the classic **Iris dataset**, loaded from:

> `iris_dataset.xlsx`

It contains 150 samples with the following features:

- `sepal_length`  
- `sepal_width`  
- `petal_length`  
- `petal_width`  
- `species`

---

## 🧪 Installation & Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/azizmuzaki4/iris-machine-learning-dashboard.git
cd iris-machine-learning-dashboard
```

### 2️⃣ Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
venv\Scripts\activate   # on Windows
source venv/bin/activate  # on macOS/Linux
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Dashboard
```bash
streamlit run "iris_machine_learning_dashboard.py"
```

Then open your browser and go to:
> http://localhost:8501

---

🌐 Live Demo

👉 [Open on Streamlit Cloud](https://iris-machine-learning-dashboard.streamlit.app/)

🧑‍💻 Author

**Aziz Muzaki**  
📍 Bekasi, Indonesia  
📧 azizmuzaki4@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/aziz-muzaki-986a75241/  
💻 GitHub: https://github.com/azizmuzaki4

---

🪪 License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute with attribution.

⭐ If you found this project helpful, please give it a star on GitHub!