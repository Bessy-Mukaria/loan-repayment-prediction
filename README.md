# Loan Prediction Data Science Project

## Overview
This project aims to develop a loan prediction model for Numida using machine learning. The primary objective is to predict whether a loan will be repaid based on historical data. The model is built using the XGBoost algorithm.

## Project Structure
The project is organized as follows:

```
loan-prediction/
├── data/
│   ├── raw/                           # Original CSV files (train, test)
│   └── processed/                     # Cleaned and transformed data files
├── notebooks/
│   └── loan_repayment_prediction.ipynb          # Jupyter notebook for analysis,EDA and model development paygrounf(Contains the approach end to end).
├── src/
│   └── loan_model.py                  # Python script for model data preparation, model training, and evaluation
├── README.md                          # Overview of the project, setup instructions
└── requirements.txt                   # Dependencies required for the project
```

## Installation
To set up the project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd loan-prediction
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment**:
   - **Windows**:
     ```bash
     .\venv\Scripts\activate
     ```
   - **Mac/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Pipeline
The main script to run the entire data processing and model training pipeline is `loan_model.py`, located in the `src` directory. To run the script, use the following command:

```bash
python src/loan_model.py
```


## Jupyter Notebook
For more interactive exploration,analysis and detailed results on the model development, you can use the Jupyter notebook provided in the `notebooks` folder:

```bash
jupyter notebook notebooks/loan_prediction.ipynb
```

The notebook allows you to experiment with the data, perform feature engineering, and visualize the results.

## Key Insights
- The XGBoost model performed well on the training data, achieving high accuracy, precision, recall, and F1-score.
- The model suffers from overfitting, as indicated by the almost perfect cross-validation scores.

## Challenges and Recommendations
### Challenges
1. **Class Imbalance**: The dataset showed an imbalance between repaid and non-repaid loans, potentially leading to biased predictions.
2. **Overfitting**: Despite regularization, the model's performance metrics suggest possible overfitting.

### Recommendations
1. **Address Class Imbalance**: Use techniques like SMOTE or adjust class weights in XGBoost.
2. **Feature Engineering**: Create new features based on domain knowledge to enhance predictive power.
3. **Hyperparameter Tuning**: Use `GridSearchCV` or `RandomizedSearchCV` to optimize model parameters.
4. **Feature Engineering**: Create new features based on domain knowledge and integration of payment related data together with the behavioral data to improve the predictive power of the model.
5. **Feature Transformation and outlier Removal**.


## Conclusion
The loan prediction model shows promising results but requires further refinement to address challenges like class imbalance and overfitting. Future work should focus on improving feature diversity, tuning hyperparameters, and making the model more interpretable use in Numida's case scenario.


