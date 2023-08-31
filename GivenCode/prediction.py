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
from joblib import dump
from joblib import load

#Import the dataset
data=pd.read_excel('Car_Purchasing_Data.xlsx')

#Display the first 5 rows of the dataset (head)
#print("first 5 rows of dataset\n",data.head())

#Display the last 5 rows of the dataset (tail)
#print("last 5 rows of dataset\n",data.tail())

#Determine the shape of the dataset (shape - Total number of rows and columns)
#print("Number of rows and columns\n",data.shape)
#print("Number of rows\n",data.shape[0])
#print("Number of columns\n",data.shape[1])

#Display the concise summary of the dataset (info)
#print(data.info())

#Check the null values in dataset (isnull)
#print(data.isnull())
# OR
#print(data.isnull().sum())

#Identify the library to plot the graph to understand the relations among the various columns
#to select the independent variables, target variables and irrelevant features.
#sns.pairplot(data)
#plt.show()


#print(data.columns)
#Create the input dataset from the original dataset by dropping the irrelevant features
# store input variables in X
X= data.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'],axis=1)
#print(X)

#Create the output dataset from the original dataset.
# store output variable in Y
Y= data['Car Purchase Amount']
#print(Y)

#Transform the input dataset into a percentage based weighted value between 0 and 1.
sc= MinMaxScaler()
X_scaled=sc.fit_transform(X)
#print(X_scaled)

#Transform the output dataset into a percentage based weighted value between 0 and 1
sc1= MinMaxScaler()
y_reshape= Y.values.reshape(-1,1)
y_scaled=sc1.fit_transform(y_reshape)
#print(Y_scaled)

# Print a few rows of the scaled input dataset (X)
#print("Scaled Input (X):")
#print(X_scaled[:5])

# Print a few rows of the scaled output dataset (y)
#print("Scaled Output (y):")
#print(y_scaled[:5])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

#print the shape of the test and train data
# print("X_train shape:", X_train.shape)
# print("X_test shape:", X_test.shape)
# print("y_train shape:", y_train.shape)
# print("y_test shape:", y_test.shape)

#print the first few rows
#head method can also be used but it only works with pandas dataframe but this can be work with numpy arrays 
# print("First 5 rows of X_train:\n", X_train[:5])
# print("First 5 rows of X_test:\n", X_test[:5])
# print("First 5 rows of y_train:\n", y_train[:5])
# print("First 5 rows of y_test:\n", y_test[:5])

#Import and Initialize the Models
lr = LinearRegression() 
svm = SVR() 
rf = RandomForestRegressor() 
gbr = GradientBoostingRegressor() 
xg =  XGBRegressor() 

#train the models using training sets
lr.fit(X_train,y_train)
svm.fit(X_train,y_train)
rf.fit(X_train,y_train)
gbr.fit(X_train,y_train) 
xg.fit(X_train,y_train)

#Prediction on the Validation/Test Data
lr_preds = lr.predict(X_test)
svm_preds = svm.predict(X_test)
rf_preds = rf.predict(X_test)
gbr_preds = gbr.predict(X_test)
xg_preds = xg.predict(X_test)

#Evaluate model performance
#RMSE is a measure of the differences between the predicted values by the model and the actual values
lr_rmse = mean_squared_error(y_test, lr_preds, squared=False)
svm_rmse = mean_squared_error(y_test, svm_preds, squared=False)
rf_rmse = mean_squared_error(y_test, rf_preds, squared=False)
gbr_rmse = mean_squared_error(y_test, gbr_preds, squared=False)
xg_rmse = mean_squared_error(y_test, xg_preds, squared=False)

#Display the evaluation results
# print(f"Linear Regression RMSE: {lr_rmse}")
# print(f"Support Vector Machine RMSE: {svm_rmse}")
# print(f"Random Forest RMSE: {rf_rmse}")
# print(f"Gradient Boosting Regressor RMSE: {gbr_rmse}")
# print(f"XGBRegressor RMSE: {xg_rmse}")

#choose the best model
model_objects = [lr, svm, rf, gbr, xg]
rmse_values = [lr_rmse, svm_rmse, rf_rmse, gbr_rmse, xg_rmse]

best_model_index = rmse_values.index(min(rmse_values))
best_model_object = model_objects[best_model_index]

#print(f"The best model is {models[best_model_index]} with RMSE: {rmse_values[best_model_index]}")

#visualize the models results
# Create a bar chart
models = ['Linear Regression', 'Support Vector Machine', 'Random Forest', 'Gradient Boosting Regressor', 'XGBRegressor']
plt.figure(figsize=(10,7))
bars = plt.bar(models, rmse_values, color=['blue', 'green', 'red', 'purple', 'orange'])

# Add RMSE values on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.00001, round(yval, 5), ha='center', va='bottom', fontsize=10)

plt.xlabel('Models')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title('Model RMSE Comparison')
plt.xticks(rotation=45)  # Rotate model names for better visibility
plt.tight_layout()
# Display the chart
plt.show()

#Retrain the model on entire dataset
lr_final = LinearRegression()
lr_final.fit(X_scaled, y_scaled)

# to know the internal working [optional]
#print("Coefficients:", lr_final.coef_)
#print("Intercept:", lr_final.intercept_)

#Save the Model
dump(best_model_object, "car_model.joblib")
#Load the model
loaded_model = load("car_model.joblib")
#print("Type of Model",type(loaded_model)) #Make sure the output is model type such as LinearRegression

# Gather user inputs
gender = int(input("Enter gender (0 for female, 1 for male): "))
age = int(input("Enter age: "))
annual_salary = float(input("Enter annual salary: "))
credit_card_debt = float(input("Enter credit card debt: "))
net_worth = float(input("Enter net worth: "))

#use the model to make predictions
X_test1= sc.transform([[gender, age, annual_salary, credit_card_debt, net_worth]])
#print(X_test1) # print just to see whether values been transformed

#Predict on new test data
pred_value= loaded_model.predict(X_test1)
print(pred_value)
print("Predicted Car_Purchase_Amount based on input:",sc1.inverse_transform(pred_value))