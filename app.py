import pandas as pd
df = pd.read_csv('data/winequality-red.csv', sep=';')
# print(df.head())

print(df.info())
print(df.describe())
print(df.isnull().sum())






