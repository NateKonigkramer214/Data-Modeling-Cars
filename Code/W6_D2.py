import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#**Varibles

filepath = r'D:\Machine_Learning_Python\Car_Purchasing_Data.xlsx'

#**Tasks W6_D1
#Import the dataset
data = pd.read_excel(filepath)
#Display the first 5 rows of the dataset (head)
# print('Display First 5 rows of the dataset!')
# print('')
# print(data.head())
# #Display the last 5 rows of the dataset (tail)
# print('Display last 5 rows of the dataset!')
# print('')
# print(data.tail())
# #Determine the ls of the dataset (shape - Total number of rows and columns)
# print('\nDataFrame Shape :', data.shape)
# print('\nNumber of rows :', data.shape[0])
# print('\nNumber of columns :', data.shape[1])
# #Display the concise summary of the dataset (info)
# print('\nDataset Info:', data.info())
# #Check the null values in dataset (isnull)
# print('\nPrints null data :', data.isnull())
# #Get overall statistics about dataset (describe)
# print('\nOverall stats about Dataset :', data.describe())
# #Identify the library to plot the graph to understand the relations among the various columns to select the independent variables, target variables and irrelevant features.
# sns.pairplot(data)
# plt.show()

#** Create the input dataset from the original dataset by dropping the irrelevant features
X= data.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'],axis=1)


#**Create the output dataset from the original dataset.
#store output variable in Y
Y= data['Car Purchase Amount']


#**Transform the input dataset into a percentage based weighted value between 0 and 1.
#! Using MinMaxScaler
scaler = MinMaxScaler()
X_Scaled = scaler.fit_transform(X)
print()


#**Transform the output dataset into a percentage based weighted value between 0 and 1
scaler1= MinMaxScaler()
#Reshape
Y_reshape= Y.values.reshape(-1,1)
Y_scaled=scaler1.fit_transform(Y_reshape)


#**Split the dataset into the training set and test set
#! The train_test_split() method is used to split our data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_Scaled, Y_scaled, test_size=0.2, random_state=42)