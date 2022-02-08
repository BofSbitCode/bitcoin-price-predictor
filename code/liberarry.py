#import liberarry 
from datetime import timedelta, date, datetime
import datetime as tm
import zipfile
import pandas as pd
import urllib.request
import progressbar
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras import models    
import math
import time as tm
from configparser import ConfigParser
import os
import json
from contextlib import redirect_stdout

def config():
    '''
    geting configuration from config.cfg file
    '''
    parser = ConfigParser()
    parser.readfp(open('../config.cfg'))
    #updateTime = parser.get("updateTime configuration", "updateTime")
    predictionDays = parser.get("runing model configuration", "predictionDays")
    #predictionDays = parser.get("predictionDays")
    doTrain = parser.get("runing model configuration", "doTrain")
    doTest = parser.get("runing model configuration", "doTest")
    modelname = parser.get("path of model and dataset configuration", "modelname")
    modelDate = parser.get("path of model and dataset configuration", "modelDate")
    baseDataSetPath = parser.get("path of model and dataset configuration", "baseDataSetPath")
    dataSetPath = parser.get("path of model and dataset configuration", "dataSetPath")
    testSize = parser.get("runing and testing model configuration","testSize")
    epochs = parser.get("runing and testing model configuration","epochs")
    splitsize = parser.get("runing and testing model configuration","splitSize")
    predictTomorrow = parser.get("runing and testing model configuration","predictTomorrow")
    return (predictTomorrow,splitsize,epochs,predictionDays,doTrain,doTest,modelname,modelDate,baseDataSetPath,dataSetPath,testSize)

def updateTime():
    '''
    geting last update date from .updateTime
    '''
    with open('../.updateTime','r') as updateTime:
        startDate = updateTime.readline()
    #calculating last update time to set for start date
    startDate = datetime.strptime(startDate, '%Y-%m-%d %H:%M:%S')
    #calculating end date
    endDate = datetime.now()
    endDate = endDate.strftime("%Y-%m-%d")
    endDate = datetime.strptime(endDate, "%Y-%m-%d")
    return startDate,endDate

def dayBetweenStartAndEnd(endDate,startDate):
    '''
    calculating day between start date and end date
    '''
    dayBetweenStartAndEnt =  endDate-startDate
    dayBetweenStartAndEnt = dayBetweenStartAndEnt.days
    #print('day between start date and end date :',dayBetweenStartAndEnt,'\nStart date = ',startDate,'and end date = ',endDate)
    return dayBetweenStartAndEnt

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

def writeDatasetcsv(totalPriceList,totalDateList,baseDataSetPath):
    '''
    writing updated data + old date in base dataset and update base dataset
    '''
    dic = {'price':totalPriceList,'time':totalDateList}
    df = pd.DataFrame(data=dic)
    df = df.drop_duplicates()
    df.to_csv(baseDataSetPath,index=False)

def writeDayByDaycsv(baseDataSetPath,dataSetPath):
    dateFrame = pd.read_csv(baseDataSetPath)
    Timestamp = list(dateFrame['time'])
    timeLengh = len(Timestamp)-(len(Timestamp)%1440)
    time = []
    timenow = datetime.now()
    for i in range(0,timeLengh,1440):
        time.append(Timestamp[i])
    time.append(Timestamp[len(Timestamp)-570])
    time.append(timenow.strftime("%Y-%m-%d")+' 14:30:00')
    Close = list(dateFrame['price'])
    price = []
    priceLengh = len(Close)-(len(Close)%1440)
    for i in range(0,priceLengh,1440):
        price.append(str(Close[i])  ) 
    price.append(Close[len(Close)-7])
    price.append(50000)
    df = pd.DataFrame(data={'price':price,'time':time})
    df.to_csv(dataSetPath,index=False)

def updateUpdateTimetxt(endDate):
    with open('../.updateTime','w') as updateTime:
        updateTime.write(str(endDate))

def splitDataset(data,siezSplit):
    lenDateFrame = len(data)  
    splitSize = math.floor((lenDateFrame*(100-siezSplit))/100)
    leftSize = lenDateFrame-splitSize
    #print('Size of split =',splitSize,'| Size left after split =',leftSize)
    return splitSize

def makeTestAndTrainPartWithPersent(data,splitSize,sizeOfTrain):
    leftDF = data.iloc[splitSize:]
    df = data.iloc[splitSize:]
    lenDateFrameLeft = len(leftDF)
    trainSize = math.floor((lenDateFrameLeft*(sizeOfTrain))/100)
    testSize = lenDateFrameLeft-trainSize
    train = leftDF.iloc[:trainSize]
    test = leftDF.iloc[trainSize:]
    #print('Size of train =',trainSize,'| Size of left =',testSize)
    return test,train

def scaler(train):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaledData = scaler.fit_transform(train['price'].values.reshape(-1,1))
    return scaledData

def makeXtrainAndyTrain(predictionDays,scaledData):
    xTrain,yTrain = [],[]
    for x in range(predictionDays, len(scaledData)):
        xTrain.append(scaledData[x-predictionDays:x,0])
        yTrain.append(scaledData[x,0])
    xTrain, yTrain = np.array(xTrain), np.array(yTrain)
    xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))
    return xTrain,yTrain

def trainModel(xTrain,yTrain,epochs,modelDate,modelname,predictionDays):
    model = Sequential()
    model.add(LSTM(units=100,return_sequences=True,input_shape=(xTrain.shape[1],1)))
    model.add(Dropout(0.25))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(units=50))
    model.add(Dropout(0.25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam',loss='mean_squared_error')
    with open('../savedModel/'+'prediction days = '+(str(predictionDays))+'/date = '+modelDate+'/name = '+modelname+'/modelSummary.info', 'w') as f:
        with redirect_stdout(f):
            model.summary()
    print('\n')
    model.fit(xTrain,yTrain,epochs=epochs,batch_size=32) 
    print('\n')
    lossPerEpoch = model.history.history['loss']
    return model,lossPerEpoch

def makeTestAndTrainPartWithTestSize(data,splitSize,testSize):
    leftDF = data.iloc[splitSize:]
    df = data.iloc[splitSize:]
    lenDateFrameLeft = len(leftDF)
    trainSize = lenDateFrameLeft-testSize
    train = leftDF.iloc[:trainSize]
    test = leftDF.iloc[trainSize:]
    #print('Size of train =',trainSize,'| Size of left =',testSize)
    return test,train

def predict(test,train,model,predictionDays):
    scaler = MinMaxScaler(feature_range=(0,1))
    actualPrices = test['price'].values
    totalDataset = pd.concat((train['price'], test['price']), axis=0)
    #totalDataset = pd.concat((train['price'],[0]), axis=0)
    modelInputs = totalDataset[len(totalDataset)-len(test)-predictionDays:].values
    modelInputs = modelInputs.reshape(-1,1)
    modelInputs = scaler.fit_transform(modelInputs)
    xTest = []
    for x in range(predictionDays, len(modelInputs)):
        xTest.append(modelInputs[x-predictionDays:x,0])
    xTest = np.array(xTest)
    xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))
    predictionPrice = model.predict(xTest)
    predictionPrice = scaler.inverse_transform(predictionPrice) 
    return (predictionPrice,actualPrices)

def makepredictcsv(predictionPrice,test,modelDate,modelname,testSize,predictionDays):
    predictionPrice = pd.DataFrame(predictionPrice, columns=['prediction price'])
    predictionPrice = predictionPrice.reset_index(drop=True)
    test = test.reset_index(drop=True)
    dataframe = pd.concat([test, predictionPrice], axis = 1)
    distance = []
    accuracy = []
    for index, row in dataframe.iterrows():
        distance.append(row['prediction price']-row['price'])
        if row['prediction price']-row['price'] < 0:
            accuracy.append(row['prediction price']*100/row['price'])
        elif row['prediction price']-row['price'] > 0:
            accuracy.append(100-((row['prediction price']*100/row['price'])-100))
        else:   
            accuracy.append(100)
    distanceAndAccuracy = pd.DataFrame({'distance':distance,'accuracy':accuracy})#([distance,accuracy],columns=['distance','accuracy']
    dataframe = pd.concat([dataframe,distanceAndAccuracy], axis=1)

    dataframe.to_csv('../savedModel/'+'prediction days = '+(str(predictionDays))+'/date = '+modelDate+'/name = '+modelname+'/prediction table | testSize = '+str(testSize)+'.csv',index=False)
    nowPrice = dataframe.iloc[0,0]
    dataframe = dataframe.iloc[1:]
    correct = 0
    false = 0
    for index, row in dataframe.iterrows():
        if row['prediction price'] > nowPrice:
            if row['price'] > nowPrice:
                correct += 1
            else:
                false += 1
        elif row['prediction price'] < nowPrice:
            if row['price'] < nowPrice:
                correct += 1
            else:
                false += 1
    return accuracy,correct,false

def BTCtoUSDT(btc,btcPrice):
    return btcPrice*btc
def USDTtoBTC(usdt,btcPrice):
    return usdt/btcPrice

def trainBot(dataframe,btc,usdt,nowPrice):
    for index, row in dataframe.iterrows():
        if row['prediction price'] > nowPrice:
            btc = btc + USDTtoBTC(usdt*0.1,nowPrice)
            usdt = usdt - (usdt*0.1)
            nowPrice = row['price']
            print(usdt,btc,'buy')
        elif row['prediction price'] < nowPrice:
            usdt = usdt + BTCtoUSDT(btc*0.1,nowPrice)
            btc = btc - (btc*0.1)
            nowPrice = row['price']
            print(usdt,btc,'sell')
    usdt = usdt + BTCtoUSDT(btc,40000)
    return usdt     

def archive(data,hour):
    selected = []
    for i in data:
        if '2013-04-01' in i:
            print('',end='')
        else:
            if hour in i:
                selected.append(i)
    with open('../dataset/archive/bitcoin2013to2022DayByDay hour = '+hour+'.csv','w') as file:
        file.write('price,time\n')
        for i in selected:
            file.write(i)

def makeTimeList():
    d0 = datetime(2018,3,21, 0, 30, 0)
    d1 = datetime(2018,3,21,23,59,59)
    secSteps = 3600.
    dt = timedelta(seconds = secSteps)
    dates = np.arange(d0, d1, dt).astype(datetime)
    timeList = []
    for date in dates: 
        time = str(date)
        time = time.split('1 ')
        timeList.append(time[1])
    return timeList

def writeLog(entry,typed,user,modelDate,modelname,predictionDays):
    now = tm.localtime()
    ms = (str(now.tm_year)+'-'+str(now.tm_mon)+'-'+str(now.tm_mday)+'  '+ str(now.tm_hour)+':'+str(now.tm_min)+':'+ str(now.tm_sec) + ' >>> ' + '[ ' + typed + ' ]' + ' : ' + user + ' | ' + entry + ' | end.\n')
    wrr = open('../savedModel/'+'prediction days = '+(str(predictionDays))+'/date = '+modelDate+'/name = '+modelname+'/btcPridictionLog.log','a')
    wrr.write(ms)
    print(ms)

def creatLogFile(modelDate,modelname,predictionDays):
    path = '../savedModel/'+'prediction days = '+(str(predictionDays))+'/date = '+modelDate+'/name = '+modelname+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    with open('../savedModel/'+'prediction days = '+(str(predictionDays))+'/date = '+modelDate+'/name = '+modelname+'/btcPridictionLog.log', 'w') as temp_file:
        ms = 'Bitcoin price predictior\nyear,month,day,hour,min,sec >>> [ type ] : user +++ ms +++ end.\n'
        temp_file.write(ms)

def infofile(avgAccuracy,maxAccuracy,minAccuracy,correct,false,datasetLenght,dataset):
    parser = ConfigParser()
    parser.read_file(open('../config.cfg'))
    predictionDays = parser.get("runing model configuration", "predictionDays")
    startDate,endDate = updateTime()
    doTrain = parser.get("runing model configuration", "doTrain")
    doTest = parser.get("runing model configuration", "doTest")
    modelname = parser.get("path of model and dataset configuration", "modelname")
    modelDate = parser.get("path of model and dataset configuration", "modelDate")
    baseDataSetPath = parser.get("path of model and dataset configuration", "baseDataSetPath")
    dataSetPath = parser.get("path of model and dataset configuration", "dataSetPath")
    description = parser.get("path of model and dataset configuration", "description")
    testSize = parser.get("runing and testing model configuration","testSize")
    epochs = parser.get("runing and testing model configuration","epochs")
    splitsize = parser.get("runing and testing model configuration","splitSize")
    unit = parser.get("runing and testing model configuration","unit")
    exchange = parser.get("runing and testing model configuration","exchange")
    timeStemp = parser.get("runing and testing model configuration","timeStemp")
    data = {
            "info":
                     {
                    "head":
                            {
                            "update time" : str(startDate.strftime("%Y-%m-%d %H:%M:%S")),
                            "name" : str(modelname),
                            "model date" :  str(endDate.strftime("%Y-%m-%d %H:%M:%S")),
                            "description" : str(description)
                            },
                    "path configuration":
                                        {
                            "base dataset path" : str(baseDataSetPath)
                            ,"dataset path" : str(dataSetPath)
                                        },
                    "traing and testing configuration":
                                                        {
                            "prediction days" : int(predictionDays),
                            "do train" : str(doTrain),
                            "do test" : str(doTest),
                            "test size" : int(testSize),
                            "epochs" : int(epochs),
                            "splot size" : int(splitsize)
                                                        },
                    "dataset info":
                                    {
                        "unit" : str(unit),
                        "exchange" : str(exchange),
                        "timeStemp" : str(timeStemp),#str(timeStemp.strftime("%H:%M:%S")),
                        "dataset lenght" : int(datasetLenght),
                        "last 5 rows" : (json.loads(dataset.head().to_json()))['price'],
                        "first 5 rows" : (json.loads(dataset.tail().to_json()))['price']
                                    }
                    },
            "accuracy":
                {
                    "avrage price accuracy" : float(avgAccuracy),
                    "maximum price accuracy" : float(maxAccuracy),
                    "minimum price accuracy" : float(minAccuracy),
                    "increases and decreases and decreases accuracy" : float(correct/(correct+false)*100),
                    "count of correct increases and decreases" : int(correct),
                    "count of flase increases and decreases" : int(false)
                },
            "model":
                {
                    "summery path": str('../savedModel/'+'prediction days = '+(str(predictionDays))+'/date = '+modelDate+'/name = '+modelname+'/modelSummary.info')
                }
            }
    data = json.dumps(data,indent=4)

    with open('../savedModel/'+'prediction days = '+(str(predictionDays))+'/date = '+modelDate+'/name = '+modelname+'/info.json', 'w') as outfile:
        outfile.write(data)
