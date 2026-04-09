import pandas as pd

# load your dataset
df = pd.read_csv("mobile_price_dataset.csv")

# see first 5 rows
print(df.head())

# basic info (very important)
print(df.info())

# check missing values
print(df.isnull().sum())
df.fillna(df.mean(), inplace=True)
df.fillna("Unknown", inplace=True)
df = pd.get_dummies(df)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# convert back to dataframe
df = pd.DataFrame(df_scaled, columns=df.columns)
import matplotlib.pyplot as plt
import seaborn as sns

# draw boxplot
sns.boxplot(data=df)
plt.show()
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
df.to_csv("cleaned_data.csv", index=False)
