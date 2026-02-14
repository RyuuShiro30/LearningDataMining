#1 Import Library 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

#2 Dataset

data = {
    'Age': [19, 21, 25, 30, 45, 50, 55, 60],
    'Income': [15000, 18000, 30000, 40000, 80000, 90000, 85000, 95000]
}

df = pd.DataFrame(data)

#3. Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

#4. Jumlah Cluster
inertia = []

for k in range(1,6):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.plot(range(1,6), inertia)
plt.xlabel("Total Tabel")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show() 

#5. Modelling KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

#6 Silhouette Score
score = silhouette_score(scaled_data, df['Cluster'])
print("Silhouette Score:", score)

#7. Visual
plt.scatter(df['Age'], df['Income'], c=df['Cluster'])
plt.xlabel("Age")
plt.ylabel("Income")
plt.title("Customer Segmentation")
plt.show()