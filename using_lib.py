import pandas as pd
import numpy as np
from sklearn.model_selection  import train_test_split
data=pd.read_csv('E:\kc_house_data.csv')
x=np.array(data['sqft_living'])
y=np.array(data['price'])
xtrain,xtest,ytrain,ytest=train_test_split(x,y)

#LinearRegression using libraries
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(xtrain.reshape(-1,1),ytrain)
ypred1=model.predict(xtest.reshape(-1,1))
mse1=mean_squared_error(ytest,ypred1)
rmse1=np.sqrt(mse1)
rmse
