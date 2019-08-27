import pandas as pd
import datetime
from matplotlib import pyplot as plt
import numpy as np


from google.colab import files
uploaded = files.upload()

def import_data():
    raw_data_df=pd.read_csv("Power-Networks-LCL.csv",header=0)
    return raw_data_df

r=import_data()
r

r['DateTime']=pd.to_datetime(r['DateTime'])
d=r.loc[:,['KWh']]
d=d.set_index(r.DateTime)
d['KWh']=pd.to_numeric(d['KWh'],downcast='float',errors='coerce')
d.head()

d.plot()
plt.show()

df=r.loc[:,['DateTime','KWh']]
df['KWh']=pd.to_numeric(df['KWh'],errors='coerce')
df=df.groupby(['DateTime']).sum().reset_index()

pd.plotting.autocorrelation_plot(df['KWh'])
plt.show()

daily=d.resample('D').sum()
mydata=daily
mydata.head()

from sklearn.preprocessing import MinMaxScaler
values=mydata['KWh'].values.reshape(-1,1)
values=values.astype('float32')
scaler=MinMaxScaler(feature_range=(0,1))
scaled=scaler.fit_transform(values)

train_size=int(len(scaled)*0.8)
test_size=len(scaled)-train_size
train,test=scaled[0:train_size,:],scaled[train_size:len(scaled),:]
print(len(train),len(test))

def create_dataset(dataset,look_back=1):
  dataX,dataY=[],[]
  for i in range(len(dataset)-look_back):
    a=dataset[i:(i+look_back),0]
    dataX.append(a)
    dataY.append(dataset[i+look_back,0])
  print(len(dataY))
  return np.array(dataX), np.array(dataY)

look_back=2
trainX,trainY=create_dataset(train,look_back)
testX,testY=create_dataset(test,look_back)

trainX=np.zeros((650,2))
trainX=np.reshape(trainX,(trainX.shape[0],1,trainX.shape[1]))
testX=np.zeros((162,2))
testX=np.reshape(testX,(testX.shape[0],1,testX.shape[1]))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

model=Sequential()
model.add(LSTM(100,input_shape=(trainX.shape[1],trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mae',optimizer='adam')
history=model.fit(trainX,trainY,epochs=300,batch_size=100,validation_data=(testX,testY),verbose=0,shuffle=False)

plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='test')
plt.legend()