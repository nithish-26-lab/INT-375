

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score
from scipy.stats import ttest_ind, shapiro

# LOAD DATASET

df = pd.read_csv("Warehouse_and_Retail_Sales.csv")

print(df.head())
print("Shape:", df.shape)

# CHECK MISSING VALUES
print("\nMissing Values:\n", df.isnull().sum())
print("Any NaN:", df.isnull().values.any())

# HANDLE MISSING VALUES

num_cols = ['RETAIL SALES','RETAIL TRANSFERS','WAREHOUSE SALES']
df[num_cols] = df[num_cols].fillna(0)

cat_cols = ['SUPPLIER','ITEM TYPE','ITEM DESCRIPTION']
df[cat_cols] = df[cat_cols].fillna('Unknown')

# REMOVE DUPLICATES

df.drop_duplicates(inplace=True)

# NUMPY ANALYSIS

print("\nMean Retail Sales:", np.mean(df['RETAIL SALES']))
print("Std Warehouse Sales:", np.std(df['WAREHOUSE SALES']))
print("Max Retail Sales:", np.max(df['RETAIL SALES']))

# SUMMARY STATISTICS

print("\nSummary Statistics:\n", df.describe())

# GROUP ANALYSIS

year_sales = df.groupby('YEAR')['WAREHOUSE SALES'].sum()
monthly_sales = df.groupby('MONTH')['WAREHOUSE SALES'].sum()

# DATA VISUALIZATION

df.columns = df.columns.str.strip()
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=num_cols)

# Distribution 
plt.figure()
sns.histplot(df['WAREHOUSE SALES'], bins=40, kde=True)
plt.title("Distribution of Warehouse Sales")
plt.show()

# Regression plot
plt.figure()
sns.regplot(x='RETAIL SALES', y='WAREHOUSE SALES', data=df, scatter_kws={'alpha':0.5})
plt.title("Retail vs Warehouse Sales")
plt.show()

# Pairplot
sns.pairplot(df[['RETAIL SALES','RETAIL TRANSFERS','WAREHOUSE SALES']])
plt.show()

# Heatmap
corr = df[['RETAIL SALES','RETAIL TRANSFERS','WAREHOUSE SALES']].corr()
plt.figure()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

#  Boxplot
top_items_list = df['ITEM TYPE'].value_counts().nlargest(10).index
plt.figure()
sns.boxplot(x='ITEM TYPE', y='RETAIL SALES', data=df[df['ITEM TYPE'].isin(top_items_list)])
plt.xticks(rotation=45)
plt.title("Retail Sales by Item Type")
plt.show()

# Bar plot
top_items = df.groupby('ITEM TYPE')['RETAIL SALES'].sum().sort_values(ascending=False).head(10)
plt.figure()
sns.barplot(x=top_items.values, y=top_items.index, palette='viridis')
plt.title("Top 10 Item Types by Sales")
plt.show()

# Monthly trend
df['MONTH'] = pd.to_numeric(df['MONTH'], errors='coerce')
monthly_sales = df.groupby('MONTH')['WAREHOUSE SALES'].sum()
plt.figure()
sns.lineplot(x=monthly_sales.index, y=monthly_sales.values, marker='o')
plt.title("Monthly Warehouse Sales Trend")
plt.show()

#Pie chart
top5 = df.groupby('ITEM TYPE')['RETAIL SALES'].sum().sort_values(ascending=False).head(5)
top5 = top5[top5 > 0]
plt.figure()
plt.pie(top5.values, labels=top5.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
plt.title("Top 5 Item Types Share")
plt.show()



# CORRELATION & COVARIANCE

print("\nCorrelation:\n", df.corr(numeric_only=True))
print("\nCovariance:\n", df.cov(numeric_only=True))



# STATISTICAL MODELLING

print("\nMean:", df['WAREHOUSE SALES'].mean())
print("Median:", df['WAREHOUSE SALES'].median())
print("Variance:", df['WAREHOUSE SALES'].var())
print("Standard Deviation:", df['WAREHOUSE SALES'].std())

#print("Skewness:", df['WAREHOUSE SALES'].skew())
#Print("Kurtosis:", df['WAREHOUSE SALES'].kurt())



# HYPOTHESIS TESTING


# Shapiro-Wilk Test (Normality)
stat, p = shapiro(df['RETAIL SALES'].sample(5000))
print("\nShapiro Test p-value:", p)

#T-Test (High vs Low Sales)
high_sales = df[df['WAREHOUSE SALES'] > df['WAREHOUSE SALES'].median()]['RETAIL SALES']
low_sales = df[df['WAREHOUSE SALES'] <= df['WAREHOUSE SALES'].median()]['RETAIL SALES']

t_stat, p_value = ttest_ind(high_sales, low_sales)

print("T-Test p-value:", p_value)

if p_value < 0.05:
    print("Significant difference between high and low sales groups")
else:
    print("No significant difference")



# OUTLIER REMOVAL

Q1 = df['RETAIL SALES'].quantile(0.25)
Q3 = df['RETAIL SALES'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df['RETAIL SALES'] >= lower) & (df['RETAIL SALES'] <= upper)]



# 14. MACHINE LEARNING

X = df[['RETAIL SALES','RETAIL TRANSFERS']]
y = df['WAREHOUSE SALES']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nR2 Score:", r2_score(y_test, y_pred))



# 15. CLASSIFICATION

df['Sales_Category'] = np.where(df['WAREHOUSE SALES'] > 100, 'High', 'Low')

X_cls = df[['RETAIL SALES','RETAIL TRANSFERS']]
y_cls = df['Sales_Category']

clf = DecisionTreeClassifier()
clf.fit(X_cls, y_cls)

# 16. SAVE FINAL DATASET

df.to_csv("final_cleaned_data.csv", index=False)
