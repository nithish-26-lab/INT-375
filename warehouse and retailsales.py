
# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind, shapiro
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score

#sns.set(style="whitegrid")

# LOAD DATASET

df = pd.read_csv("Warehouse_and_Retail_Sales.csv")

print("First 5 rows:\n", df.head())
print("\nShape:", df.shape)

# DATA CLEANING

print("\nMissing Values:\n", df.isnull().sum())

num_cols = ['RETAIL SALES','RETAIL TRANSFERS','WAREHOUSE SALES']
df[num_cols] = df[num_cols].fillna(0)

cat_cols = ['SUPPLIER','ITEM TYPE','ITEM DESCRIPTION']
df[cat_cols] = df[cat_cols].fillna('Unknown')

df.drop_duplicates(inplace=True)

# Feature Engineering
df['TOTAL SALES'] = df['RETAIL SALES'] + df['WAREHOUSE SALES']

# DESCRIPTIVE STATISTICS
print("\nSummary Statistics:\n", df.describe())

print("\nMean Retail Sales:", np.mean(df['RETAIL SALES']))
print("Std Warehouse Sales:", np.std(df['WAREHOUSE SALES']))


# VISUALIZATION (EDA)

# Distribution
sns.histplot(df['WAREHOUSE SALES'], kde=True)
plt.title("Distribution of Warehouse Sales")
plt.show()

# KDE
sns.kdeplot(df['RETAIL SALES'], fill=True)
plt.title("KDE Plot - Retail Sales")
plt.show()

# Pairplot
sns.pairplot(df[['RETAIL SALES','RETAIL TRANSFERS','WAREHOUSE SALES']])
plt.show()

# Boxplot
sns.boxplot(data=df[['RETAIL SALES','RETAIL TRANSFERS','WAREHOUSE SALES']])
plt.title("Boxplot for Outlier Detection")
plt.show()

# Scatter
sns.scatterplot(x='RETAIL SALES', y='WAREHOUSE SALES', data=df)
plt.title("Retail vs Warehouse Sales")
plt.show()

# Regression Plot
sns.regplot(x='RETAIL SALES', y='WAREHOUSE SALES', data=df)
plt.title("Regression Plot")
plt.show()

# Heatmap
sns.heatmap(df[['RETAIL SALES','RETAIL TRANSFERS','WAREHOUSE SALES']].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Year-wise Trend
year_sales = df.groupby('YEAR')['WAREHOUSE SALES'].sum()
plt.plot(year_sales.index, year_sales.values, marker='o')
plt.title("Year-wise Warehouse Sales Trend")
plt.xlabel("Year")
plt.ylabel("Sales")
plt.show()

# Top Item Types
top_items = df.groupby('ITEM TYPE')['RETAIL SALES'].sum().nlargest(10)
sns.barplot(x=top_items.values, y=top_items.index)
plt.title("Top 10 Item Types")
plt.show()

# OUTLIER REMOVAL (IQR)
Q1 = df['RETAIL SALES'].quantile(0.25)
Q3 = df['RETAIL SALES'].quantile(0.75)
IQR = Q3 - Q1

df = df[(df['RETAIL SALES'] >= Q1 - 1.5*IQR) &
        (df['RETAIL SALES'] <= Q3 + 1.5*IQR)]

# CORRELATION
print("\nCorrelation:\n", df.corr(numeric_only=True))

# STATISTICAL TESTING

# Shapiro-Wilk Test (Normality)
stat, p = shapiro(df['RETAIL SALES'].sample(5000))
print("\nShapiro Test p-value:", p)

# T-Test (Compare two variables)
t_stat, p_val = ttest_ind(df['RETAIL SALES'], df['WAREHOUSE SALES'])
print("T-Test p-value:", p_val)


# LINEAR REGRESSION MODEL

X = df[['RETAIL SALES','RETAIL TRANSFERS']]
y = df['WAREHOUSE SALES']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))


#CLASSIFICATION (BONUS)

df['Sales_Category'] = np.where(df['WAREHOUSE SALES'] > 100, 'High', 'Low')

X_cls = df[['RETAIL SALES','RETAIL TRANSFERS']]
y_cls = df['Sales_Category']

clf = DecisionTreeClassifier()
clf.fit(X_cls, y_cls)


#SAVE FINAL DATA

df.to_csv("final_cleaned_data.csv", index=False)


