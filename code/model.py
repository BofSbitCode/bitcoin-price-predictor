from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras import models
def Model(xTrain):
    model = Sequential()
    model.add(LSTM(units=100,return_sequences=True,input_shape=(xTrain.shape[1],1)))
    model.add(Dropout(0.25))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(units=50))
    model.add(Dropout(0.25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam',loss='mean_squared_error')
    return model