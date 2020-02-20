import pandas as pd
import numpy as np
from sklearn.model_selection  import train_test_split
data=pd.read_csv('...\kc_house_data.csv')
x=np.array(data['sqft_living'])
y=np.array(data['price'])
xtrain,xtest,ytrain,ytest=train_test_split(x,y)

#LinearRegression python code
xmean=np.mean(xtrain)
ymean=np.mean(ytrain)
numerator=np.sum((xtrain-xmean)*(ytrain-ymean))
denom=np.sum((xtrain-xmean)**2)
m=numerator/denom
b=ymean-m*xmean
ypred=m*xtest+b
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(ytest,ypred)
rmse=np.sqrt(mse)
print(rmse) #gives the mean squared error 
