
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import pandas as pd

# Function to train the model and save both the model and the scaler
def train_model(data_path, model_path='xgb_model.pkl', scaler_path='scaler.pkl'):
    """
    This function trains an XGBoost model using the entire training dataset and saves the model and scaler.
    
    Parameters:
    - data_path: Path to the CSV file containing the training data.
    - model_path: Path where the trained model will be saved.
    - scaler_path: Path where the scaler will be saved.
    """
    # Load the training dataset
    data = pd.read_csv(data_path)
    
    # Prepare features and target variable (next change prediction)
    X = data[['macd_60_120', 'macd_60_480', 'macd_60_960', 'rsi_60', 'volume_1h_8']]
    y = data['instrument_value_change_gross_pct']
    
    # Initialize and fit the scaler on the training data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize and train the XGBoost model for regression (next change)
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Save the trained model and scaler for later use
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

# Function to optimize the model using hyperparameter tuning with GridSearchCV
def optimize_model_grid_search(data_path, model_path='optimized_xgb_model.pkl', scaler_path='scaler.pkl'):
    """
    This function performs hyperparameter optimization using GridSearchCV, trains the best model, and saves it.
    """
    # Load the dataset
    data = pd.read_csv(data_path)
    X = data[['macd_60_120', 'macd_60_480', 'macd_60_960', 'rsi_60', 'volume_1h_8']]
    y = data['instrument_value_change_gross_pct']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define parameter grid for XGBoost
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 10],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.3]
    }
    
    # XGBoost model
    model = XGBRegressor(random_state=42)
    
    # Grid Search for Hyperparameter Tuning
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_scaled, y)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    
    # Save the best model and scaler
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"Optimized model saved to {model_path}")

# Function to load the model and make predictions for both next change and direction of change
def predict_model(test_data, model_path='optimized_xgb_model.pkl', scaler_path='scaler.pkl'):
    """
    This function takes test_data (DataFrame), automatically excludes the 'timestamp' and 'instrument_value_change_gross_pct' columns, 
    and performs predictions for both next change (regression) and direction of change (classification).
    
    Parameters:
    - test_data: The DataFrame containing the test features, including 'timestamp' and 'instrument_value_change_gross_pct'.
    """
    # Load the saved model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Extract the feature columns (ignoring the 'timestamp' and 'instrument_value_change_gross_pct' columns)
    X_new = test_data[['macd_60_120', 'macd_60_480', 'macd_60_960', 'rsi_60', 'volume_1h_8']]
    
    # Scale the new data using the loaded scaler
    X_new_scaled = scaler.transform(X_new)
    
    # Predict the next change (continuous value for regression)
    next_change_prediction = model.predict(X_new_scaled)
    
    # Predict the direction of the change (1 for positive change, 0 for negative/zero change)
    direction_prediction = (next_change_prediction > 0).astype(int)
    
    # Attach the predictions to the test data with the timestamp for reference
    results_df = test_data[['timestamp']].copy()
    results_df['next_change_prediction'] = next_change_prediction
    results_df['direction_prediction'] = direction_prediction
    
    return results_df

# Function to evaluate the model performance on unseen test data (with true target values)
def evaluate_model(test_data, model_path='optimized_xgb_model.pkl', scaler_path='scaler.pkl'):
    """
    This function takes test_data (DataFrame with true target values), automatically excludes the 'timestamp' column,
    and evaluates the model on both next change (regression) and direction of change (classification).
    
    Parameters:
    - test_data: The DataFrame containing the features and true target values.
    """
    # Load the saved model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Extract the feature columns (ignoring the 'timestamp' and 'instrument_value_change_gross_pct' columns)
    X_new = test_data[['macd_60_120', 'macd_60_480', 'macd_60_960', 'rsi_60', 'volume_1h_8']]
    
    # True target values for evaluation (next change)
    y_true = test_data['instrument_value_change_gross_pct']
    
    # Scale the new data using the loaded scaler
    X_new_scaled = scaler.transform(X_new)
    
    # Predict the next change
    y_pred = model.predict(X_new_scaled)
    
    # Evaluate the regression task (next change)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    
    # Predict the direction of the change (1 for positive, 0 for negative/zero)
    direction_pred = (y_pred > 0).astype(int)
    direction_true = (y_true > 0).astype(int)
    
    # Evaluate the classification task (direction of change)
    accuracy = accuracy_score(direction_true, direction_pred)
    
    # Output the results
    print(f"MAE on unseen data (next change): {mae:.4f}")
    print(f"RMSE on unseen data (next change): {rmse:.4f}")
    print(f"Accuracy on unseen data (direction of change): {accuracy:.4f}")
    
    return y_pred, direction_pred
