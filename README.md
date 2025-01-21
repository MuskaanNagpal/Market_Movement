
Stock Price Prediction and Trend Analysis
This repository contains the implementation of a machine learning pipeline to predict stock price changes and classify directional trends using technical indicators. The project leverages cutting-edge machine learning models like XGBoost, Random Forest, and LightGBM, combined with feature engineering and interpretability tools for actionable insights.

Objectives
Predict stock price changes (regression task) with high precision.
Classify directional trends in stock price movement (classification task) for actionable insights.
Identify key features driving predictions and understand their impact using interpretability techniques like SHAP.

Key Features
Feature Engineering:
Extracted meaningful financial indicators such as MACD, RSI, and trading volume to represent market behavior.
Calculated interaction terms between features using polynomial expansion to capture complex relationships.

Machine Learning Models:
Implemented and optimized XGBoost, Random Forest, and LightGBM models.
Hyperparameter tuning with GridSearchCV for model optimization.

Evaluation Metrics:
Regression performance measured using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
Classification performance evaluated with accuracy and confusion matrices.

Model Interpretability:
Used SHAP (SHapley Additive Explanations) to identify the most important features influencing predictions.

Time-Series Analysis:
Explored trends, seasonality, and volatility in stock prices using decomposition and autocorrelation plots.
Approach

Data Preprocessing:
Cleaned and transformed time-series data.
Scaled features using StandardScaler for optimal model performance.

Exploratory Data Analysis (EDA):
Conducted univariate and bivariate analysis to identify feature distributions and relationships.
Used visualization tools like heatmaps, scatter plots, and regression plots to analyze feature-target dependencies.

Modeling:
Developed separate pipelines for regression (next price change) and classification (direction of price movement).
Evaluated models with 5-fold cross-validation to ensure robust performance on unseen data.

Hyperparameter Tuning:
Used GridSearchCV to optimize parameters like learning rate, max depth, and n_estimators for the best-performing models.

Results
Achieved a test RMSE of 0.0010 and MAE of 0.0005 for stock price prediction.
Classification models demonstrated 95% accuracy in predicting price movement direction.
SHAP analysis revealed trading volume and RSI as the most influential predictors.
