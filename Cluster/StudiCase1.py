import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv('customer_electronic.csv')

X = df[['Age', 'Income', 'Monthly_Spending']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)

df['Cluster'] = kmeans.fit_predict(X_scaled)
print(df)

df.to_csv('customer_electronic_clustered.csv', index=False)

plt.scatter(df['Age'], df['Income'], c=df['Cluster'])

plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Customer Segmentation')
plt.show()