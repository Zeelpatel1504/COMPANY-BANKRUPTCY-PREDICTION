# COMPANY-BANKRUPTCY-PREDICTION
# **Company Bankruptcy Prediction**

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Dataset Details](#dataset-details)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Feature Engineering](#feature-engineering)
5. [Model Selection & Training](#model-selection--training)
6. [Cross-Validation Strategy](#cross-validation-strategy)
7. [Model Performance & Results](#model-performance--results)
8. [Technologies Used](#technologies-used)
9. [How to Run the Project](#how-to-run-the-project)
10. [Project Contributors](#project-contributors)
11. [Acknowledgments](#acknowledgments)
12. [Contact Information](#contact-information)

---

## **Project Overview**
The objective of this project is to develop a machine learning model that accurately predicts a company’s bankruptcy risk using financial indicators. This project is part of the **CSP 571 - Data Preparation and Analysis** course and aims to help financial institutions and stakeholders evaluate company stability in real-time.

### **Importance of Bankruptcy Prediction**
- Helps **stakeholders** identify financial risks early.
- Assists in **resource allocation** and **risk mitigation** strategies.
- Enhances **financial stability** and contributes to economic resilience.

---

## **Dataset Details**
- **Source**: [Kaggle - Company Bankruptcy Prediction](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction/data)
- **Target Variable**: `Y - Bankrupt?` (Binary: 1 = Bankrupt, 0 = Not Bankrupt)
- **Key Features**: Financial ratios and company performance indicators.
- **Data Preprocessing**:
  - Handling **missing values**, **duplicates**, and ensuring dataset stability.
  - **Feature selection** based on correlation analysis.

---

## **Exploratory Data Analysis (EDA)**
- **Feature Correlation Analysis**: Identify relationships between financial indicators.
- **Impact on Target Variable**: Determining which financial metrics significantly impact bankruptcy prediction.
- **Independence Assumption Validation**: Ensuring features do not exhibit high collinearity.
- **Data Visualization Techniques**:
  - **Correlation Plots**: Visualizing financial data relationships.
  - **Dimensionality Reduction**:
    - **Principal Component Analysis (PCA)**
    - **UMAP (Uniform Manifold Approximation and Projection)**
    - **t-SNE (t-Distributed Stochastic Neighbor Embedding)**

---

## **Feature Engineering**
- **Scaling & Normalization**: Standardizing numerical features.
- **Handling Class Imbalance**:
  - Data balancing from `6599:220` to `6599:6599`.
  - **SMOTE (Synthetic Minority Over-sampling Technique)** used.
- **Feature Transformation**: Applying log transformations where necessary.

---

## **Model Selection & Training**
The following machine learning and deep learning models were trained and tested:
1. **Logistic Regression**
2. **Random Forest Classifier**
3. **XGBoost (Extreme Gradient Boosting)**
4. **K-Nearest Neighbors (KNN)**
5. **Deep Learning Models**

**Evaluation Metrics:**
- **Accuracy**
- **Precision & Recall**
- **F1-score**
- **ROC-AUC Score**

---

## **Cross-Validation Strategy**
To ensure model robustness, we used two cross-validation techniques:
1. **Stratified k-Fold Cross-Validation**:
   - Handles imbalanced datasets effectively.
   - Ensures each fold maintains the same class distribution.
2. **Time Series Split** (if applicable):
   - Used for financial data over time.
   - Ensures training on past data and testing on future data.

---

## **Model Performance & Results**
### **Key Insights**
- **XGBoost & Deep Learning achieved the highest accuracy (98%)**.
- **Data Balancing** significantly improved model performance.
- **Strong Stability**: Models maintained consistency across training and testing datasets.

---

## **Technologies Used**
- **Programming Language**: Python
- **Libraries & Frameworks**:
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `matplotlib` & `seaborn`
  - `XGBoost`
  - `TensorFlow` & `Keras`

---

## **How to Run the Project**
### **Prerequisites**
Ensure you have Python installed along with the necessary dependencies:
```bash
pip install pandas numpy scikit-learn xgboost tensorflow keras matplotlib seaborn
```
### **Steps to Run**
1. **Clone the repository**
   ```bash
   git clone <repository_url>
   cd bankruptcy_prediction
   ```
2. **Run the Data Preprocessing Script**
   ```bash
   python data_preprocessing.py
   ```
3. **Train and Evaluate Models**
   ```bash
   python train_models.py
   ```
4. **View Results & Visualizations**
   - The final reports and visualizations will be generated in the `results/` directory.

---

## **Project Contributors**
This project was conducted by **Group 21** from CSP 571 - Data Preparation and Analysis:
- **Zeel Patel**
- **Sundar Machani**

---

## **Acknowledgments**
Special thanks to **Illinois Institute of Technology** and **Kaggle** for providing the dataset and course resources.

---

## **Contact Information**
For any queries, feel free to reach out:
- **Zeel Patel** – zpatel@hawk.iit.edu
- **Sundar Machani** – smachani@hawk.iit.edu



