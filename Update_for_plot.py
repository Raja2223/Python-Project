import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from scipy.stats import probplot
from statsmodels.graphics.gofplots import qqplot
#To load data
path="C:/Users/rajak/JAVA VS CODE/raja/EV_DataSet_PY.csv"
df=pd.read_csv(path)
# Box plot of EV percentage by vehicle type
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Vehicle Primary Use', y='Percent Electric Vehicles', 
            palette='Set2')
plt.title('Distribution of EV Percentage by Vehicle Type')
plt.xlabel('Vehicle Primary Use')
plt.ylabel('EV Percentage')
plt.tight_layout()
plt.show()

# Donut chart of BEVs vs PHEVs
bev_total = df['Battery Electric Vehicles (BEVs)'].sum()
phev_total = df['Plug-In Hybrid Electric Vehicles (PHEVs)'].sum()

plt.figure(figsize=(8, 8))
plt.pie([bev_total, phev_total], labels=['BEVs', 'PHEVs'], 
        autopct='%1.1f%%', startangle=90, colors=['darkgreen', 'limegreen'])
centre_circle = plt.Circle((0,0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title('Proportion of BEVs vs PHEVs in Dataset')
plt.show()

# Scatter plot: EV Total vs Non-Electric Vehicle Total
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Non-Electric Vehicle Total', y='Electric Vehicle (EV) Total', 
                hue='State', alpha=0.7)
plt.title('EV vs Non-EV Vehicles by State')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Non-Electric Vehicles (log scale)')
plt.ylabel('Electric Vehicles (log scale)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Heatmap of correlation matrix
plt.figure(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of EV Dataset')
plt.tight_layout()
plt.show()

# Bar plot of top 10 states by EV percentage
state_ev = df.groupby('State')['Percent Electric Vehicles'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
state_ev.plot(kind='bar', color='green')
plt.title('Top 10 States by Average EV Percentage')
plt.ylabel('Average EV Percentage')
plt.xlabel('State')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Histogram of EV percentages
plt.figure(figsize=(10, 6))
plt.hist(df['Percent Electric Vehicles'], bins=30, color='teal', edgecolor='black')
plt.title('Distribution of EV Percentages')
plt.xlabel('Percentage of Electric Vehicles')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()