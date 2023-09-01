# Import statements (unchanged)
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump, load

# Factory function to create models
def create_model(model_name):
    if model_name == 'Linear Regression':
        return LinearRegression()
    elif model_name == 'Support Vector Machine':
        return SVR()
    elif model_name == 'Random Forest':
        return RandomForestRegressor()
    elif model_name == 'Gradient Boosting Regressor':
        return GradientBoostingRegressor()
    elif model_name == 'XGBRegressor':
        return XGBRegressor()
    else:
        raise ValueError("Invalid model name")

# Load the dataset
def load_data(filepath):
    return pd.read_excel(filepath)

# Preprocess the data
def preprocess_data(data):
    X = data.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis=1)
    Y = data['Car Purchase Amount']
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    scaler1 = MinMaxScaler()
    Y_reshape = Y.values.reshape(-1, 1)
    Y_scaled = scaler1.fit_transform(Y_reshape)
    
    return X_scaled, Y_scaled, scaler, scaler1

# Split the dataset
def split_data(X_scaled, Y_scaled, test_size=0.2, random_state=42):
    return train_test_split(X_scaled, Y_scaled, test_size=test_size, random_state=random_state)

# Train models
def train_models(X_train, y_train):
    lr = LinearRegression() 
    svm = SVR() 
    rf = RandomForestRegressor() 
    gbr = GradientBoostingRegressor() 
    xg = XGBRegressor() 
    
    models = [lr, svm, rf, gbr, xg]
    model_names = ['Linear Regression', 'Support Vector Machine', 'Random Forest', 'Gradient Boosting Regressor', 'XGBRegressor']
    trained_models = {}
    
    for model, name in zip(models, model_names):
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

# Evaluate models
def evaluate_models(models, X_test, y_test, scaler1):
    rmse_values = {}
    
    for name, model in models.items():
        predictions = model.predict(X_test)
        rmse = mean_squared_error(y_test, predictions, squared=False)
        rmse_values[name] = rmse
        print(f"{name} RMSE: {rmse}")
    
    return rmse_values

# Plot model performance
def plot_model_performance(rmse_values):
    plt.figure(figsize=(12, 8))
    plt.bar(rmse_values.keys(), rmse_values.values(), color='blue')
    plt.xlabel('Models')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title('Model RMSE Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Save the best model
def save_best_model(models, rmse_values):
    best_model_name = min(rmse_values, key=rmse_values.get)
    best_model = models[best_model_name]
    dump(best_model, "best_model.joblib")

# Predict outcome with new data
def predict_new_data(input_data, scaler, loaded_model, scaler1):
    scaled_input = scaler.transform([input_data])
    pred_value = loaded_model.predict(scaled_input)
    original_pred_value = scaler1.inverse_transform(pred_value)
    return original_pred_value


# Main function with factory pattern
def main():
    filepath = r'D:\Machine_Learning_Python\Car_Purchasing_Data.xlsx'
    data = load_data(filepath)
    X_scaled, Y_scaled, scaler, scaler1 = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X_scaled, Y_scaled)
    
    model_names = ['Linear Regression', 'Support Vector Machine', 'Random Forest', 'Gradient Boosting Regressor', 'XGBRegressor']
    
    trained_models = {}
    for name in model_names:
        model = create_model(name)
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    rmse_values = evaluate_models(trained_models, X_test, y_test, scaler1)
    plot_model_performance(rmse_values)
    save_best_model(trained_models, rmse_values)
    
    # Example input for prediction
    input_data = [0, 30, 60000, 2000, 40000]  # Replace with actual input
    model_name = min(rmse_values, key=rmse_values.get)
    loaded_model = trained_models[model_name]
    prediction = predict_new_data(input_data, scaler, loaded_model, scaler1)
    print(f"Predicted Car Purchase Amount using {model_name}:", prediction)

if __name__ == "__main__":
    main()
