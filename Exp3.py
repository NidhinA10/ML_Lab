import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("datasets/diabetes.csv")
'''
data = load_diabetes(as_frame=True)
df = data.frame
'''
X = df.drop("target",axis=1)
Y = df["target"]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
lr_model = Pipeline([
	("scaler",StandardScaler()),
	("model",LinearRegression())
	])
lr_model.fit(X_train,Y_train)
y_pred_lr = lr_model.predict(X_test)
lr_mse = mean_squared_error(Y_test, y_pred_lr)
lr_r2 = r2_score(Y_test, y_pred_lr)

ridge_pipe = Pipeline([
	("scaler",StandardScaler()),
	("model",Ridge())
	])
ridge_params = {
	'model__alpha' : [0.01,0.1,1,10,100]
	}
ridge_cv = GridSearchCV(
	ridge_pipe,
	ridge_params,
	cv = 5,
	scoring = 'neg_mean_squared_error'
	)
ridge_cv.fit(X_train,Y_train)
best_ridge = ridge_cv.best_estimator_
y_pred_ridge = best_ridge.predict(X_test)
ridge_mse = mean_squared_error(Y_test, y_pred_ridge)
ridge_r2 = r2_score(Y_test, y_pred_ridge)

lasso_pipe = Pipeline([
	("scaler",StandardScaler()),
	("model",Lasso(max_iter = 10000))
	])
lasso_params = {
	'model__alpha' : [0.001,0.01,0.1,1,10]
	}
lasso_cv = GridSearchCV(
	lasso_pipe,
	lasso_params,
	cv=5,
	scoring = 'neg_mean_squared_error'
	)
lasso_cv.fit(X_train,Y_train)
best_lasso = lasso_cv.best_estimator_
y_pred_lasso = best_lasso.predict(X_test)
lasso_mse = mean_squared_error(Y_test, y_pred_lasso)
lasso_r2 = r2_score(Y_test, y_pred_lasso)

print("Best Ridge Alpha :",ridge_cv.best_params_['model__alpha'])
print("Best Lasso Alpha :",lasso_cv.best_params_['model__alpha'])
results = pd.DataFrame({
	"Model" : ["Linear Regression","Ridge Regression", "Lasso Regression"],
	"MSE" : [lr_mse,ridge_mse,lasso_mse],
	"R2_Score" : [lr_r2,ridge_r2,lasso_r2]
	})
print("\nPerformance Comparison\n")
print(results)
