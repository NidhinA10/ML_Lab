import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay
cols = ["sepal_length","sepal_width","petal_length","petal_width","species"]
df = pd.read_csv("datasets/iris.csv",names=cols).dropna()
X = df[["petal_length","petal_width"]]
Y = (df["species"] == "Iris-setosa").astype(int)
'''
data = load_iris()
X = data.data[:, 2:4]
Y=(data.target ==0).astype(int)
'''
scaler = StandardScaler()
X=scaler.fit_transform(X)
model = SVC(kernel = "linear",C = 1)
model.fit(X,Y)
pred = model.predict(X)
print("Accuracy : ",accuracy_score(Y,pred))
DecisionBoundaryDisplay.from_estimator(
	model,
	X,
	plot_method="contour",
	levels=[-1,0,1],
	colors="k",
	linestyles=["--","-","--"])
plt.scatter(X[:,0],X[:,1],c =Y,edgecolors='k')
plt.scatter(
	model.support_vectors_[ :,0],
	model.support_vectors_[:,1],
	s=100,
	facecolors="none",
	edgecolors="red")
plt.xlabel("Petal Length (scaled)")
plt.ylabel("Petal Width (scaled)")
plt.title("Linear SVM: Iris Classification")
plt.show()
