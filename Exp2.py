import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

df = pd.read_csv("datasets/auto-mpg.csv")
df = df[["displacement","mpg"]].replace("?",np.nan)
df = df.dropna()
df = df.astype(float)

X = df[["displacement"]]
Y = df["mpg"]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
lr_model = Pipeline([
	("scaler",StandardScaler()),
	("model",LinearRegression())
	])
lr_model.fit(X_train,Y_train)
y_pred_lr = lr_model.predict(X_test)
print("Linear Regression")
print("MSE : ",mean_squared_error(Y_test,y_pred_lr))
print("R2_Score : ",r2_score(Y_test,y_pred_lr))
plt.figure(figsize=(10,6))
plt.scatter(X,Y,alpha=0.5,label="Actual Data")
X_Curve = pd.DataFrame({"displacement" : np.linspace(
		X["displacement"].min(),
		X["displacement"].max(),
		300)
		})
plt.plot(X_Curve["displacement"],lr_model.predict(X_Curve),linewidth=2,label="LinearRegression")

degrees=[2,3,4,5]
for degree in degrees:
	poly_model = Pipeline([
		("polynomial",PolynomialFeatures(degree=degree)),
		("linear",LinearRegression())
		])
	poly_model.fit(X_train,Y_train)
	prediction = poly_model.predict(X_test)
	print("Polynomial Degree : ",degree)
	print("MSE : ",mean_squared_error(Y_test,prediction))
	print("R2_Score : ",r2_score(Y_test,prediction))
	plt.plot(X_Curve["displacement"],poly_model.predict(X_Curve),linewidth=2,label=f"Polynomial Degree {degree}")
plt.xlabel("Displacement")
plt.ylabel("Miles Per Gallon")
plt.title("Auto MPG")
plt.legend()
plt.grid(True)
plt.show()
	

