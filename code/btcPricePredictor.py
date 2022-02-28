from liberarry import bitcoinPricePredictor
predictor = bitcoinPricePredictor()
predictor.creatLogFile()
predictor.writeLog('importing libraries','start')
from datetime import timedelta, date, datetime
import zipfile
import pandas as pd
import urllib.request
import progressbar
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import models  
predictor.writeLog('importing libraries','successful')
startDate,endDate = predictor.updateTime()
dayBetweenStartAndEnt = predictor.dayBetweenStartAndEnd(endDate,startDate)
if dayBetweenStartAndEnt != 0 :
    totalPriceList,totalDateList = predictor.update()
    predictor.writeDatasetcsv(totalPriceList,totalDateList)
    predictor.writeDayByDaycsv()
    predictor.updateUpdateTimetxt(endDate)
data = predictor.read(True)
splitSize = predictor.splitDataset(data)
test,train = predictor.makeTestAndTrainPartWithTestSize(data)
scaledData = predictor.scaler(train)
xTrain,yTrain = predictor.makeXtrainAndyTrain(scaledData)
if predictor.doTrain == 'True' :
    model = predictor.trainModel(xTrain,yTrain)    
if predictor.doTest == 'True' :
    predictionPrice,actualPrices = predictor.predict(test,train,model)
    if predictor.predictTomorrow == 'True':
        predictor.predictTomorrow(predictionPrice)
    accuracy,correct,false = predictor.makepredictcsv(predictionPrice,test)
    avgAccuracy,minAccuracy,maxAccuracy = predictor.acc(accuracy,correct,false)
    data = predictor.read(False)
    datasetLenght = len(data)
    predictor.infofile(avgAccuracy,maxAccuracy,minAccuracy,correct,false,datasetLenght,data)
