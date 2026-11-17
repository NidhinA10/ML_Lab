import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
fashion = fetch_openml("Fashion-MNIST",version=1,as_frame=False)
X = fashion.data
y = fashion.target.astype(int)
X = X[:12000]
y = y[:12000]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train / 255.0
X_test = X_test / 255.0
"""
train = pd.read_csv("datasets/fashion_train.csv")
test = pd.read_csv("datasets/fashion_test.csv")
X_train = train.drop("label", axis=1).iloc[:10000] / 255.0
y_train = train["label"].iloc[:10000]
X_test = test.drop("label", axis=1).iloc[:2000] / 255.0
y_test = test["label"].iloc[:2000]
"""
k_values = [1, 3, 5, 7, 9]
accuracies = []
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    accuracies.append(acc)
    print("K =", k, "Accuracy =", acc)
plt.plot(k_values, accuracies, marker="o")
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.title("KNN: Accuracy vs K")
plt.xticks(k_values)
plt.grid(True)
plt.show()

