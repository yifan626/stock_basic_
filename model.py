import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import pickle

# Reading the data
df = pd.read_csv(r'https://raw.githubusercontent.com/yifan626/stock_basic_/main/netflix.csv')
print(df.head())
used_features = ["High", "Low", "Open", "Volume"]
X = df[used_features]
y = df["Close_Last"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0,)

regr = linear_model.LinearRegression()

regr.fit(X_train, y_train)

stocks_y_pred = regr.predict(X_test)

pickle.dump(regr, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))
