import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.pipeline import Pipeline
"""
df = pd.read_csv("datasets/california_housing.csv")
df["AveRooms"] = df["total_rooms"] / df["households"]
"""
data = fetch_california_housing(as_frame=True)
df = data.frame
X = df[["AveRooms"]]
#median_house_value would be for read_csv
Y = df["MedHouseVal"] / 100000
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
lr_model = Pipeline([ 
	("scaler",StandardScaler()),
	("model",LinearRegression())
	])
lr_model.fit(X_train,Y_train)
y_pred_lr = lr_model.predict(X_test)
print("Normal Equation")
print("MSE : ",mean_squared_error(Y_test,y_pred_lr))
print("R2_Score : ",r2_score(Y_test,y_pred_lr))

gd_model = Pipeline([
	("scaler",StandardScaler()),
	("model",SGDRegressor(max_iter=1000,tol=1e-3,random_state=42))
	])
gd_model.fit(X_train,Y_train)
y_pred_gd = gd_model.predict(X_test)
print("Gradient Descent")
print("MSE : ",mean_squared_error(Y_test,y_pred_gd))
print("R2_Score : ",r2_score(Y_test,y_pred_gd))

idx=X_test["AveRooms"].argsort()
plt.scatter(X_test["AveRooms"],Y_test,s=10)
plt.plot(X_test["AveRooms"].iloc[idx],y_pred_gd[idx],color="green",label="Gradient Descent")
plt.plot(X_test["AveRooms"].iloc[idx],y_pred_lr[idx],color="red",label="Normal Equation")
plt.grid(True)
plt.legend()
plt.xlabel("Average Rooms Per Dwelling")
plt.ylabel("Median Housing Value")
plt.title("California Housing Using Function")
plt.show()
