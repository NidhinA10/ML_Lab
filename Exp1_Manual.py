import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score

df=pd.read_csv("datasets/california_housing.csv")
df["AveRooms"] = df["total_rooms"] / df["households"]
X = df[["AveRooms"]]
Y = df["median_house_value"]/100000
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
Xtr = np.c_[np.ones(len(X_train)),X_train]
Xte = np.c_[np.ones(len(X_test)),X_test]
print("Gradient Descent")
theta = np.zeros(2)
etha = 0.01
for i in range(1000):
	error = Xtr @ theta - Y_train
	theta -= etha * (Xtr.T @ error) / len(Y_train)
y_pred_gd = Xte @ theta
print("MSE : ",mean_squared_error(Y_test,y_pred_gd))
print("R2_Score : ",r2_score(Y_test,y_pred_gd)) 
print("\n Normal Equation")
theta = np.linalg.pinv(Xtr.T @ Xtr) @ Xtr.T @ Y_train
y_pred_ne = Xte @ theta
print("MSE : ",mean_squared_error(Y_test,y_pred_ne))
print("R2_Score : ",r2_score(Y_test,y_pred_ne)) 
idx = np.argsort(X_test[:,0])
plt.scatter(X_test[:,0],Y_test,s=10)
plt.plot(X_test[idx,0],y_pred_gd[idx],"r", label="Gradient Descent",linewidth=5)
plt.plot(X_test[idx,0],y_pred_ne[idx],"g", label = "Normal Equation")
plt.xlabel("Average Rooms Per Dwelling")
plt.ylabel("Median Housing Value")
plt.legend()
plt.title("California Housing")
plt.show()
