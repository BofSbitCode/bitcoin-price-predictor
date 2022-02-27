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
from os import getlogin
import json
from contextlib import redirect_stdout

class bitcoinPricePredictor():
    parser = ConfigParser()
    parser.readfp(open('../config.cfg'))
    predictionDays = int(parser.get("runing model configuration", "predictionDays"))
    doTrain = parser.get("runing model configuration", "doTrain")
    doTest = parser.get("runing model configuration", "doTest")
    modelname = parser.get("path of model and dataset configuration", "modelname")
    modelDate = parser.get("path of model and dataset configuration", "modelDate")
    baseDataSetPath = parser.get("path of model and dataset configuration", "baseDataSetPath")
    dataSetPath = parser.get("path of model and dataset configuration", "dataSetPath")
    testSize = int(parser.get("runing and testing model configuration","testSize"))
    epochs = int(parser.get("runing and testing model configuration","epochs"))
    splitngSize = int(parser.get("runing and testing model configuration","splitSize"))
    predictTomorrow = parser.get("runing and testing model configuration","predictTomorrow")
    description = parser.get("path of model and dataset configuration", "description")
    unit = parser.get("runing and testing model configuration","unit")
    exchange = parser.get("runing and testing model configuration","exchange")
    timeStemp = parser.get("runing and testing model configuration","timeStemp")
    user = str(getlogin())

    def read(self,parse):
        self.writeLog('reading day by day dataset','reading')
        data = pd.read_csv(self.dataSetPath,index_col='time',parse_dates=parse)
        self.writeLog('reading day by day dataset','successful')
        return data

    def update(self):
        def daterange(start_date, end_date):
            for n in range(int ((end_date - start_date).days)):
                yield start_date + timedelta(n)
        
        predictor.writeLog('reading base dataset','reading')
        data = pd.read_csv(baseDataSetPath)
        predictor.writeLog('reading base dataset','successful')

        predictor.writeLog('making price and time list from base dataset','making')
        totalPriceList = list(data['price'])
        totalDateList = list(data['time'])
        predictor.writeLog('making price and time list from base dataset','successful')

        predictor.writeLog('downloding new data','downloding')
        bar = progressbar.ProgressBar(maxval=dayBetweenStartAndEnt, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        count = 0
        for single_date in daterange(startDate, endDate):
            time = single_date.strftime("%Y-%m-%d")
            url = 'https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT-1m-'+time+'.zip'
            zipPath = '../dataset/zip/'+time+'.zip'
            csvPath = '../dataset/csv/BTCUSDT-1m-'+time+'.csv'
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
        return totalPriceList,totalDateList

    def config(self):
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
        usr = getlogin()
        return (predictTomorrow,doTrain,doTest,baseDataSetPath,dataSetPath,testSize)

    def creatLogFile(self):
        path = '../savedModel/'+'prediction days = '+(str(self.predictionDays))+'/date = '+self.modelDate+'/name = '+self.modelname+'/'
        if not os.path.exists(path):
            os.makedirs(path)
        with open('../savedModel/'+'prediction days = '+(str(self.predictionDays))+'/date = '+self.modelDate+'/name = '+self.modelname+'/btcPridictionLog.log', 'w') as temp_file:
            ms = 'Bitcoin price predictior\nyear,month,day,hour,min,sec >>> [ type ] : user +++ ms +++ end.\n'
            temp_file.write(ms)

    def updateTime(self):
        self.writeLog('reading .updateTime','reading')
        with open('../.updateTime','r') as updateTime:
            startDate = updateTime.readline()
        startDate = datetime.strptime(startDate, '%Y-%m-%d %H:%M:%S')
        endDate = datetime.now()
        endDate = endDate.strftime("%Y-%m-%d")
        endDate = datetime.strptime(endDate, "%Y-%m-%d")
        ms = str('start date = '+endDate.strftime('%Y-%m-%d %H:%M:%S')+' end date = '+startDate.strftime('%Y-%m-%d %H:%M:%S'))
        self.writeLog(ms,'return')
        self.writeLog('reading .updateTime','successful')
        return startDate,endDate

    def writeLog(self,entry,typed):
        now = tm.localtime()
        user = str(self.user)
        ms = (str(now.tm_year)+'-'+str(now.tm_mon)+'-'+str(now.tm_mday)+'  '+ str(now.tm_hour)+':'+str(now.tm_min)+':'+ str(now.tm_sec) + ' >>> ' + '[ ' + typed + ' ]' + ' : ' + user + ' | ' + entry + ' | end.\n')
        wrr = open('../savedModel/'+'prediction days = '+(str(self.predictionDays))+'/date = '+self.modelDate+'/name = '+self.modelname+'/btcPridictionLog.log','a')
        wrr.write(ms)
        print(ms)
    
    def dayBetweenStartAndEnd(self,endDate,startDate):
        self.writeLog('calcutaing day between start date and end date','calcutaing')
        dayBetweenStartAndEnt =  endDate-startDate
        dayBetweenStartAndEnt = dayBetweenStartAndEnt.days
        ms = str('day between start date and end date = '+str(dayBetweenStartAndEnt))
        self.writeLog(ms,'return')
        self.writeLog('calcutaing day between start date and end date','successful')

        return dayBetweenStartAndEnt

    def writeDatasetcsv(self,totalPriceList,totalDateList):
        self.writeLog('downloding new data','successful')
        self.writeLog('updating base dataset','updating')
        dic = {'price':totalPriceList,'time':totalDateList}
        df = pd.DataFrame(data=dic)
        df = df.drop_duplicates()
        df.to_csv(self.baseDataSetPath,index=False)
        self.writeLog('updating base dataset','successful')

    def writeDayByDaycsv(self):
        self.writeLog('making day by day dataset','making')
        dateFrame = pd.read_csv(self.baseDataSetPath)
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
        df.to_csv(self.dataSetPath,index=False)
        self.writeLog('making day by day dataset','successful')
    
    def updateUpdateTimetxt(self,endDate):
        self.writeLog('updating .updateTime','updating')
        with open('../.updateTime','w') as updateTime:
            updateTime.write(str(endDate))
        self.writeLog('updating .updateTime','successful')

    def splitDataset(self,data):
        self.writeLog('spliting dataset to smaler part','spliting')
        lenDateFrame = len(data)  
        splitSize = math.floor((lenDateFrame*(100-(self.splitngSize)))/100)
        leftSize = lenDateFrame-splitSize
        ms = str('smaler size after split = '+str(splitSize))
        self.writeLog(ms,'return')
        self.writeLog('spliting dataset to smaler part','successful')
        return splitSize    

    def makeTestAndTrainPartWithPersent(self,data):
        leftDF = data.iloc[self.splitSize:]
        df = data.iloc[self.splitSize:]
        lenDateFrameLeft = len(leftDF)
        trainSize = math.floor((lenDateFrameLeft*(self.testSize))/100)
        testSize = lenDateFrameLeft-trainSize
        train = leftDF.iloc[:trainSize]
        test = leftDF.iloc[trainSize:]
        #print('Size of train =',trainSize,'| Size of left =',testSize)
        return test,train

    def scaler(self,train):
        self.writeLog('scaling data in 0,1','scaling')
        scaler = MinMaxScaler(feature_range=(0,1))
        scaledData = scaler.fit_transform(train['price'].values.reshape(-1,1))
        self.writeLog('scaling data in 0,1','successful')
        return scaledData

    def makeXtrainAndyTrain(self,scaledData):
        self.writeLog('making xTrain and yTain','making')
        xTrain,yTrain = [],[]
        for x in range(self.predictionDays, len(scaledData)):
            xTrain.append(scaledData[x-self.predictionDays:x,0])
            yTrain.append(scaledData[x,0])
        xTrain, yTrain = np.array(xTrain), np.array(yTrain)
        xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))
        self.writeLog('making xTrain and yTrain','successful')
        return xTrain,yTrain

    def trainModel(self,xTrain,yTrain):
        self.writeLog('training start','train')
        model = Sequential()
        model.add(LSTM(units=100,return_sequences=True,input_shape=(xTrain.shape[1],1)))
        model.add(Dropout(0.25))
        model.add(LSTM(units=50,return_sequences=True))
        model.add(Dropout(0.25))
        model.add(LSTM(units=50))
        model.add(Dropout(0.25))
        model.add(Dense(units=1))
        model.compile(optimizer='adam',loss='mean_squared_error')
        with open('../savedModel/'+'prediction days = '+(str(self.predictionDays))+'/date = '+self.modelDate+'/name = '+self.modelname+'/modelSummary.info', 'w') as f:
            with redirect_stdout(f):
                model.summary()
        print('\n')
        model.fit(xTrain,yTrain,epochs=self.epochs,batch_size=32) 
        print('\n')
        self.writeLog('training finishd','successful')
        lossPerEpoch = model.history.history['loss']
        self.writeLog('saving model','saving')
        model.save('../savedModel/'+'prediction days = '+(str(self.predictionDays))+'/date = '+self.modelDate+'/name = '+self.modelname+'/model.h5')
        self.writeLog('saving model','successful')
        plt.plot(range(len(lossPerEpoch)),lossPerEpoch)
        self.writeLog('ploting loss history of trainig model','plot')
        plt.savefig('../savedModel/'+'prediction days = '+(str(self.predictionDays))+'/date = '+self.modelDate+'/name = '+self.modelname+'/lossplot.png')
        self.writeLog('ploting loss history of trainig model','successful')
        return model

    def makeTestAndTrainPartWithTestSize(self,data):
        self.writeLog('spliting dataset to train part and test part','spliting')
        leftDF = data.iloc[self.splitngSize:]
        df = data.iloc[self.splitngSize:]
        lenDateFrameLeft = len(leftDF)
        trainSize = lenDateFrameLeft-self.testSize
        train = leftDF.iloc[:trainSize]
        test = leftDF.iloc[trainSize:]
        ms = str('test size after split = '+str(len(test)))
        self.writeLog(ms,'return')
        self.writeLog('spliting dataset to train part and test part','successful')
        return test,train

    def predict(self,test,train,model):
        self.writeLog('predicting price','predicting')
        self.writeLog('loading saved model','loading')
        model=models.load_model('../savedModel/'+'prediction days = '+(str(self.predictionDays))+'/date = '+self.modelDate+'/name = '+self.modelname+'/model.h5')
        self.writeLog('loading saved model','successful')
        self.writeLog('predicting price','test')
        scaler = MinMaxScaler(feature_range=(0,1))
        actualPrices = test['price'].values
        totalDataset = pd.concat((train['price'], test['price']), axis=0)
        #totalDataset = pd.concat((train['price'],[0]), axis=0)
        modelInputs = totalDataset[len(totalDataset)-len(test)-self.predictionDays:].values
        modelInputs = modelInputs.reshape(-1,1)
        modelInputs = scaler.fit_transform(modelInputs)
        xTest = []
        for x in range(self.predictionDays, len(modelInputs)):
            xTest.append(modelInputs[x-self.predictionDays:x,0])
        xTest = np.array(xTest)
        xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))
        predictionPrice = model.predict(xTest)
        predictionPrice = scaler.inverse_transform(predictionPrice) 
        self.writeLog('predicting price','successful')
        self.writeLog('ploting prediction price and actual price','plot')
        plt.plot(actualPrices,marker='o',color='blue',linestyle='dashed',linewidth = 0.5,markerfacecolor='gray', markersize=1,label='Actual prices')#label='Actual Prices',
        plt.plot(predictionPrice,marker='o',color='green',linestyle='dashed',linewidth = 0.5,markerfacecolor='gray', markersize=1) #label='Predited Prices'
        plt.title('BTC price prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        self.writeLog('ploting prediction price and actual price','successful')
        self.writeLog('saving prediction price and actual price plot','saving')
        plt.savefig('../savedModel/'+'prediction days = '+(str(self.predictionDays))+'/date = '+self.modelDate+'/name = '+self.modelname+'/prediction plot | testSize = '+str(self.testSize)+'.png')
        return (predictionPrice,actualPrices)

    def predictTomorrow(self,predictionPrice):
        startDate,endDate = self.updateTime()
        self.writeLog('calcutaing tomorrow price','calcutaing')
        tomorrowPrice = int(predictionPrice[len(predictionPrice)-1])
        self.writeLog('calcutaing tomorrow price','successful')
        ms = 'tomorrow price = '+str(tomorrowPrice)
        self.writeLog(ms,'return')
        self.writeLog('saving tomorrow price prediction','saving')
        with open('../tomorrowPrice','w') as file:
            file.write(str(endDate.strftime("%Y-%m-%d"))+' 14:30:00(UTC) >>>>> '+str(tomorrowPrice))
        self.writeLog('saving tomorrow price prediction','successful')        

    def makepredictcsv(self,predictionPrice,test):
        self.writeLog('making predict csv and making accuracy list','making')
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

        dataframe.to_csv('../savedModel/'+'prediction days = '+(str(self.predictionDays))+'/date = '+self.modelDate+'/name = '+self.modelname+'/prediction table | testSize = '+str(self.testSize)+'.csv',index=False)
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
        self.writeLog('making predict csv and making accuracy list','successful')
        return accuracy,correct,false

    def acc(self,accuracy,correct,false):
        self.writeLog('calcutaing accuracy','calcutaing')
        totalAccuracy = 0
        for i in accuracy:
            totalAccuracy = totalAccuracy + i
        avgAccuracy = totalAccuracy/len(accuracy)
        maxAccuracy = max(accuracy)
        minAccuracy = min(accuracy)
        ms = str('avrage accuracy = '+str(avgAccuracy)+' max accuracy = '+str(maxAccuracy)+' min accuracy = '+str(minAccuracy)+' increases and decreases and decreases accuracy = '+str(correct/(correct+false))+' count of correct increases and decreases = '+str(correct)+' count of flase increases and decreases = '+str(false))
        self.writeLog(ms,'return')
        self.writeLog('calcutaing accuracy','successful')
        return avgAccuracy,minAccuracy,maxAccuracy

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

    def api():
        print('start')
        import datetime as dt
        import numpy as np
        import pandas as pd
        from zipfile import ZipFile
        zf = ZipFile('btcusd.csv.zip')
        cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        dfs = pd.concat({text_file.filename.split('.')[0]: pd.read_csv(zf.open(text_file.filename),usecols=cols)for text_file in zf.infolist()if text_file.filename.endswith('.csv')})
        print(dfs)
        df = dfs.droplevel(1).reset_index().rename(columns={'index':'ticker'})
        df = df[df['ticker'].str.contains('usd')]
        df['date'] = df['time']
        #df['date'] = pd.to_datetime(df['time'], unit=None)
        df = df.sort_values(by=['date','ticker'])
        df = df.drop(columns='time')
        df = df.set_index(['date','ticker'])
        #df = df['2020-07-01':'2020-12-31']c
        print(df)
        bars1m = df
        #bars1m = bars1m.reset_index().set_index('date').groupby('ticker').resample('60').last().droplevel(0)
        #bars1m = bars1m.reset_index().set_index('date').groupby('ticker').last().droplevel(0)
        bars1m.loc[:, bars1m.columns[:-1]] = bars1m[bars1m.columns[:-1]].ffill()
        bars1m.loc[:, 'volume'] = bars1m['volume'].fillna(value=0.0)
        bars1m = bars1m.reset_index().set_index(['date','ticker'])
        print(bars1m)
        bars1m.to_csv('crypto-price-data.csv')
        print('start')
        import datetime as dt
        import numpy as np
        import pandas as pd
        from zipfile import ZipFile
        zf = ZipFile('btcusd.csv.zip')
        cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        dfs = pd.concat({text_file.filename.split('.')[0]: pd.read_csv(zf.open(text_file.filename),usecols=cols)for text_file in zf.infolist()if text_file.filename.endswith('.csv')})
        print(dfs)
        df = dfs.droplevel(1).reset_index().rename(columns={'index':'ticker'})
        df = df[df['ticker'].str.contains('usd')]
        #df['date'] = df['time']
        df['date'] = pd.to_datetime(df['time'], unit='ms')
        df = df.sort_values(by=['date','ticker'])
        df = df.drop(columns='time')
        df = df.set_index(['date','ticker'])
        #df = df['2020-07-01':'2020-12-31']c
        print(df)
        bars1m = df
        bars1m = bars1m.reset_index().set_index('date').groupby('ticker').resample('60').last().droplevel(0)
        #bars1m = bars1m.reset_index().set_index('date').groupby('ticker').last().droplevel(0)
        bars1m.loc[:, bars1m.columns[:-1]] = bars1m[bars1m.columns[:-1]].ffill()
        bars1m.loc[:, 'volume'] = bars1m['volume'].fillna(value=0.0)
        bars1m = bars1m.reset_index().set_index(['date','ticker'])
        print(bars1m)
        bars1m('close')(i) == df('close')(i)

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
        
    def dataset():
        data = open('../dataset/-bitcoin2013to2022.csv','r')
        data = data.readlines()
        nowdate = '2022-02-01 11:15:00'
        time = makeTimeList()
        bar = progressbar.ProgressBar(maxval=len(time), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        count = 0
        for i in range (len(time)):
            archive(data,time[i])
            count = count + 1
            bar.update(count)   

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

    def infofile(self,avgAccuracy,maxAccuracy,minAccuracy,correct,false,datasetLenght,dataset):
        startDate,endDate = self.updateTime()
        data = {
                "info":
                        {
                        "head":
                                {
                                "update time" : str(startDate.strftime("%Y-%m-%d %H:%M:%S")),
                                "name" : str(self.modelname),
                                "model date" :  str(endDate.strftime("%Y-%m-%d %H:%M:%S")),
                                "description" : str(self.description)
                                },
                        "path configuration":
                                            {
                                "base dataset path" : str(self.baseDataSetPath)
                                ,"dataset path" : str(self.dataSetPath)
                                            },
                        "traing and testing configuration":
                                                            {
                                "prediction days" : int(self.predictionDays),
                                "do train" : str(self.doTrain),
                                "do test" : str(self.doTest),
                                "test size" : int(self.testSize),
                                "epochs" : int(self.epochs),
                                "splot size" : int(self.splitngSize)
                                                            },
                        "dataset info":
                                        {
                            "unit" : str(self.unit),
                            "exchange" : str(self.exchange),
                            "timeStemp" : str(self.timeStemp),#str(timeStemp.strftime("%H:%M:%S")),
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
                        "summery path": str('../savedModel/'+'prediction days = '+(str(self.predictionDays))+'/date = '+self.modelDate+'/name = '+self.modelname+'/modelSummary.info')
                    }
                }
        data = json.dumps(data,indent=4)

        with open('../savedModel/'+'prediction days = '+(str(self.predictionDays))+'/date = '+self.modelDate+'/name = '+self.modelname+'/info.json', 'w') as outfile:
            outfile.write(data)
