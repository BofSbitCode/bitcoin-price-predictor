#import liberarry 
from liberarry import *
from os import getlogin
usr = getlogin()
predictTomorrow,splitsize,epochs,predictionDays,doTrain,doTest,modelname,modelDate,baseDataSetPath,dataSetPath,testSize = config()
creatLogFile(modelDate,modelname,predictionDays)
predictionDays = int(predictionDays)
testSize = int(testSize)
epochs = int(epochs)
splitsize = int(splitsize)
#predictTomorrow = 'True'

writeLog('importing libraries','start',usr,modelDate,modelname,str(predictionDays))
from datetime import timedelta, date, datetime
import zipfile
import pandas as pd
import urllib.request
import progressbar
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import models  
writeLog('importing libraries','successful',usr,modelDate,modelname,str(predictionDays))

writeLog('reading .updateTime','reading',usr,modelDate,modelname,str(predictionDays))
startDate,endDate = updateTime()
ms = str('start date = '+endDate.strftime('%Y-%m-%d %H:%M:%S')+' end date = '+startDate.strftime('%Y-%m-%d %H:%M:%S'))
writeLog(ms,'return',usr,modelDate,modelname,str(predictionDays))
writeLog('reading .updateTime','successful',usr,modelDate,modelname,str(predictionDays))

writeLog('calcutaing day between start date and end date','calcutaing',usr,modelDate,modelname,str(predictionDays))
dayBetweenStartAndEnt = dayBetweenStartAndEnd(endDate,startDate)
ms = str('day between start date and end date = '+str(dayBetweenStartAndEnt))
writeLog(ms,'return',usr,modelDate,modelname,str(predictionDays))
writeLog('calcutaing day between start date and end date','successful',usr,modelDate,modelname,str(predictionDays))

if dayBetweenStartAndEnt != 0 :
    def daterange(start_date, end_date):
        for n in range(int ((end_date - start_date).days)):
            yield start_date + timedelta(n)
    
    writeLog('reading base dataset','reading',usr,modelDate,modelname,str(predictionDays))
    data = pd.read_csv(baseDataSetPath)
    writeLog('reading base dataset','successful',usr,modelDate,modelname,str(predictionDays))

    writeLog('making price and time list from base dataset','making',usr,modelDate,modelname,str(predictionDays))
    totalPriceList = list(data['price'])
    totalDateList = list(data['time'])
    writeLog('making price and time list from base dataset','successful',usr,modelDate,modelname,str(predictionDays))

    writeLog('downloding new data','downloding',usr,modelDate,modelname,str(predictionDays))
    bar = progressbar.ProgressBar(maxval=dayBetweenStartAndEnt, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    count = 0
    for single_date in daterange(startDate, endDate):
        time = single_date.strftime("%Y-%m-%d")
        url = 'https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT-1m-'+time+'.zip'
        zipPath = '../dataset/zip/'+time+'.zip'
        csvPath = '../dataset/csv/BTCUSDT-1m-'+time+'.csv'
        #req = requests.get(url, allow_redirects=True)
        req = urllib.request.urlopen(url)
        content = req.read()
        with open(zipPath, 'wb') as file:
            file.write(content)
        with zipfile.ZipFile(zipPath, 'r') as zip_ref:
            zip_ref.extractall('../dataset/csv/')
        with open(csvPath,'+r') as btc:
            btc.writelines('time,open,high,low,close,volume,Close time,Quote asset volume,Number of trades,Taker buy base asset volume,Taker buy quote asset volume,Ignore,')
            btc.close()
        df = pd.read_csv(csvPath)
        df.drop(['open','high','low','volume','Close time','Close time','Quote asset volume','Number of trades','Taker buy base asset volume','Taker buy quote asset volume','Ignore'], axis = 1, inplace = True)
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df.drop(df.columns[2], axis=1, inplace=True)
        df.drop(df.columns[2], axis=1, inplace=True)
        newRow = pd.DataFrame({'close':0,'time':time+' 00:00:00'},index=[0])
        df = pd.concat([newRow, df]).reset_index(drop = True)
        dfPrice = list(df['close']) 
        dfDate = list(df['time'])
        totalPriceList = totalPriceList + dfPrice
        totalDateList = totalDateList + dfDate
        df.to_csv(csvPath,index=False)
        count = count + 1
        bar.update(count)   
    bar.finish()
    writeLog('downloding new data','successful',usr,modelDate,modelname,str(predictionDays))
    writeLog('updating base dataset','updating',usr,modelDate,modelname,str(predictionDays))
    writeDatasetcsv(totalPriceList,totalDateList,baseDataSetPath)
    writeLog('updating base dataset','successful',usr,modelDate,modelname,str(predictionDays))

    writeLog('making day by day dataset','making',usr,modelDate,modelname,str(predictionDays))
    writeDayByDaycsv(baseDataSetPath,dataSetPath)
    writeLog('making day by day dataset','successful',usr,modelDate,modelname,str(predictionDays))

    writeLog('updating .updateTime','updating',usr,modelDate,modelname,str(predictionDays))
    updateUpdateTimetxt(endDate)
    writeLog('updating .updateTime','successful',usr,modelDate,modelname,str(predictionDays))

writeLog('reading day by day dataset','reading',usr,modelDate,modelname,str(predictionDays))
data = pd.read_csv(dataSetPath,index_col='time',parse_dates=True)
writeLog('reading day by day dataset','successful',usr,modelDate,modelname,str(predictionDays))

writeLog('spliting dataset to smaler part','spliting',usr,modelDate,modelname,str(predictionDays))
ms = str('smaler size after split = '+str(splitsize))
writeLog(ms,'return',usr,modelDate,modelname,str(predictionDays))
splitSize = splitDataset(data,splitsize)
writeLog('spliting dataset to smaler part','successful',usr,modelDate,modelname,str(predictionDays))

writeLog('spliting dataset to train part and test part','spliting',usr,modelDate,modelname,str(predictionDays))
ms = str('test size after split = '+str(testSize))
writeLog(ms,'return',usr,modelDate,modelname,str(predictionDays))
test,train = makeTestAndTrainPartWithTestSize(data,splitSize,testSize)
writeLog('spliting dataset to train part and test part','successful',usr,modelDate,modelname,str(predictionDays))

writeLog('scaling data in 0,1','scaling',usr,modelDate,modelname,str(predictionDays))
scaler = MinMaxScaler(feature_range=(0,1))
scaledData = scaler.fit_transform(train['price'].values.reshape(-1,1))
writeLog('scaling data in 0,1','successful',usr,modelDate,modelname,str(predictionDays))

writeLog('making xTrain and yTain','making',usr,modelDate,modelname,str(predictionDays))
xTrain,yTrain = makeXtrainAndyTrain(predictionDays,scaledData)
writeLog('making xTrain and yTrain','successful',usr,modelDate,modelname,str(predictionDays))

if doTrain == 'True' :
    writeLog('training start','train',usr,modelDate,modelname,str(predictionDays))
    model,lossPerEpoch = trainModel(xTrain,yTrain,epochs,modelDate,modelname,predictionDays)    
    writeLog('training finishd','successful',usr,modelDate,modelname,str(predictionDays))
    #loss_per_epoch = model.history.history['loss']
    writeLog('saving model','saving',usr,modelDate,modelname,str(predictionDays))
    model.save('../savedModel/'+'prediction days = '+(str(predictionDays))+'/date = '+modelDate+'/name = '+modelname+'/model.h5')
    writeLog('saving model','successful',usr,modelDate,modelname,str(predictionDays))
    plt.plot(range(len(lossPerEpoch)),lossPerEpoch)
    writeLog('ploting loss history of trainig model','plot',usr,modelDate,modelname,str(predictionDays))
    plt.savefig('../savedModel/'+'prediction days = '+(str(predictionDays))+'/date = '+modelDate+'/name = '+modelname+'/lossplot.png')
    writeLog('ploting loss history of trainig model','successful',usr,modelDate,modelname,str(predictionDays))

if doTest == 'True' :
    writeLog('loading saved model','loading',usr,modelDate,modelname,str(predictionDays))
    model=models.load_model('../savedModel/'+'prediction days = '+(str(predictionDays))+'/date = '+modelDate+'/name = '+modelname+'/model.h5')
    writeLog('loading saved model','successful',usr,modelDate,modelname,str(predictionDays))
    writeLog('predicting price','test',usr,modelDate,modelname,str(predictionDays))
    predictionPrice,actualPrices = predict(test,train,model,predictionDays)
    if predictTomorrow == 'True':
        writeLog('calcutaing tomorrow price','calcutaing',usr,modelDate,modelname,str(predictionDays))
        tomorrowPrice = int(predictionPrice[len(predictionPrice)-1])
        writeLog('calcutaing tomorrow price','successful',usr,modelDate,modelname,str(predictionDays))
        ms = 'tomorrow price = '+str(tomorrowPrice)
        writeLog(ms,'return',usr,modelDate,modelname,str(predictionDays))
        writeLog('saving tomorrow price prediction','saving',usr,modelDate,modelname,str(predictionDays))
        with open('../tomorrowPrice','w') as file:
            file.write(str(endDate.strftime("%Y-%m-%d"))+' 14:30:00(UTC) >>>>> '+str(tomorrowPrice))
        writeLog('saving tomorrow price prediction','successful',usr,modelDate,modelname,str(predictionDays))
    writeLog('predicting price','successful',usr,modelDate,modelname,str(predictionDays))
    writeLog('ploting prediction price and actual price','plot',usr,modelDate,modelname,str(predictionDays))
    plt.plot(actualPrices,marker='o',color='blue',linestyle='dashed',linewidth = 0.5,markerfacecolor='gray', markersize=1,label='Actual prices')#label='Actual Prices',
    plt.plot(predictionPrice,marker='o',color='green',linestyle='dashed',linewidth = 0.5,markerfacecolor='gray', markersize=1) #label='Predited Prices'
    plt.title('BTC price prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    writeLog('ploting prediction price and actual price','successful',usr,modelDate,modelname,str(predictionDays))
    writeLog('saving prediction price and actual price plot','saving',usr,modelDate,modelname,str(predictionDays))
    plt.savefig('../savedModel/'+'prediction days = '+(str(predictionDays))+'/date = '+modelDate+'/name = '+modelname+'/prediction plot | testSize = '+str(testSize)+'.png')
    writeLog('saving prediction price and actual price plot','successful',usr,modelDate,modelname,str(predictionDays))
    writeLog('making predict csv and making accuracy list','making',usr,modelDate,modelname,str(predictionDays))
    accuracy,correct,false = makepredictcsv(predictionPrice,test,modelDate,modelname,testSize,predictionDays)
    writeLog('making predict csv and making accuracy list','successful',usr,modelDate,modelname,str(predictionDays))
    writeLog('calcutaing accuracy','calcutaing',usr,modelDate,modelname,str(predictionDays))
    totalAccuracy = 0
    for i in accuracy:
        totalAccuracy = totalAccuracy + i
    avgAccuracy = totalAccuracy/len(accuracy)
    maxAccuracy = max(accuracy)
    minAccuracy = min(accuracy)
    ms = str('avrage accuracy = '+str(avgAccuracy)+' max accuracy = '+str(maxAccuracy)+' min accuracy = '+str(minAccuracy)+' increases and decreases and decreases accuracy = '+str(correct/(correct+false))+' count of correct increases and decreases = '+str(correct)+' count of flase increases and decreases = '+str(false))
    writeLog(ms,'return',usr,modelDate,modelname,str(predictionDays))   
    writeLog('calcutaing accuracy','successful',usr,modelDate,modelname,str(predictionDays))
    #fAndDAccuracy = 123
    data = pd.read_csv(dataSetPath,index_col='time')
    datasetLenght = len(data)
    infofile(avgAccuracy,maxAccuracy,minAccuracy,correct,false,datasetLenght,data)
