import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score,f1_score
"""
df = pd.read_csv("datasets/20newsgroups.csv")
X = df["text"].fillna("")
Y = df["target"]
"""
X, Y = fetch_20newsgroups(subset="all", return_X_y=True)
vectorizer = CountVectorizer(stop_words="english",max_features=5000)
X = vectorizer.fit_transform(X)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
print("Multinomial NB")
model_mnb = MultinomialNB()
model_mnb.fit(X_train,Y_train)
y_pred_mnb=model_mnb.predict(X_test)
print("Accuracy Score : ",accuracy_score(Y_test,y_pred_mnb))
print("F1_Score : ",f1_score(Y_test,y_pred_mnb,average="weighted"))
print("Bernoulli NB")
model_bnb = BernoulliNB(binarize=0.0)
model_bnb.fit(X_train,Y_train)
y_pred_bnb=model_bnb.predict(X_test)
print("Accuracy Score : ",accuracy_score(Y_test,y_pred_bnb))
print("F1_Score : ",f1_score(Y_test,y_pred_bnb,average="weighted"))
