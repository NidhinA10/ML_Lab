import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("datasets/winequality-red.csv",sep=";")
X = df.drop("quality",axis=1)
Y=df["quality"]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
architectures = [(16,),(32,),(32,16),(64,32,16)]
results=[]
for layers in architectures:
	model = MLPClassifier(hidden_layer_sizes = layers, max_iter = 500, random_state=42)
	model.fit(X_train,Y_train)
	pred= model.predict(X_test)
	accuracy=accuracy_score(Y_test,pred)
	results.append(accuracy)
	print("Layers :",layers, "Accuracy :",accuracy)
plt.bar([str(a) for a in architectures],results)
plt.xlabel("Hidden Layers")
plt.ylabel("Accuracy")
plt.title("MLP")
plt.show()
