import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

#**Varibles

filepath = r'D:\Machine_Learning_Python\Car_Purchasing_Data.xlsx'

#Tasks W6_D1
#**Import the dataset
data = pd.read_excel(filepath)
#**Display the first 5 rows of the dataset (head)
print('Display First 5 rows of the dataset!')
print('')
print(data.head())
#**Display the last 5 rows of the dataset (tail)
print('Display last 5 rows of the dataset!')
print('')
print(data.tail())
#**Determine the ls of the dataset (shape - Total number of rows and columns)
print('\nDataFrame Shape :', data.shape)
print('\nNumber of rows :', data.shape[0])
print('\nNumber of columns :', data.shape[1])
#**Display the concise summary of the dataset (info)
print('\nDataset Info:', data.info())
#**Check the null values in dataset (isnull)
print('\nPrints null data :', data.isnull())
#**Get overall statistics about dataset (describe)
print('\nOverall stats about Dataset :', data.describe())
#**Identify the library to plot the graph to understand the relations among the various columns to select the independent variables, target variables and irrelevant features.
sns.pairplot(data)
plt.show()




