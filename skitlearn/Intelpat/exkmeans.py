
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt


diabetes = load_diabetes()
X = diabetes.data       
y = diabetes.target     





scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


wss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
    kmeans.fit(X_scaled)
    wss.append(kmeans.inertia_)


plt.plot(range(1, 11), wss)
plt.title("Elbow Method for Diabetes Data")
plt.xlabel("Number of clusters")
plt.ylabel("WSS (Inertia)")
plt.show()


kmeans = KMeans(n_clusters=3, init='k-means++', random_state=0)
kmeans.fit(X_scaled)
labels = kmeans.labels_


from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(y, labels)
print(f"Adjusted Rand Index: {ari:.4f}")
