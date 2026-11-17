import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

df = pd.read_csv("datasets/digits.csv")
X = df.drop("target",axis=1)
X = StandardScaler().fit_transform(X)
inertia = []
silhouette = []
K = range(2,16)
for k in K:
	model = KMeans(n_clusters = k,random_state=42, n_init=10)
	model.fit(X)
	labels = model.predict(X)
	inertia.append(model.inertia_)
	silhouette.append(silhouette_score(X,labels))
plt.subplot(1,2,1)
plt.plot(K,inertia,marker="o")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.subplot(1,2,2)
plt.plot(K,silhouette,marker="o")
plt.xlabel("K")
plt.ylabel("Silhouette")
plt.title("Silhouette Analysis")
plt.tight_layout()
plt.show()
