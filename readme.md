# Bitcoin price predictor 
predict price of bitcoin with python and tensorflow 
<br><br>

## Get dataset
Get dataset from this drive link : <a href = 'drive.google.com'>click hare for google drive link</a><br>
or this kaggle link : <a href = 'kaggle.com'>click hare for google drive link</a><br>
p.s : In kaggle dataset first you should edit dataset for using in model but drive link has already been edited.
after downloding dataset put it in <code>dataset/</code> or you can put it anywhere and copy the path of dataset in <code>config.cfg baseDataSetPath</code><br><br>

## Creat your own model
Creat your own model by edit <code>config.cfg</code> and then run <code>code/btcPricePredictor.py</code>
<br>
model directory :  <code>savedModel/prediction days = modelPredictionDays/date = modelDate/name = modelNdame</code><br><br>

## Calculating the price of tomorrow
in <code>config.cfg</code> predictTomorrow should be "True". then you can run <code>code/btcPricePredictor.py</code> and see the result and see price of tomorrow in <code>tomorrowPrice</code><br><br>

## See result
You can see the log in <code>btcPredctionLog.log</code>, information of model in <code>info.json</code>, <code>lossplot.png</code>, <code>modelSummary.info</code> and <code>prediction plot</code> and <code>prediction table</code> csv in model directory
<br><br>

## result i got
you can see models i train in <code>savedModel</code>
for example you can see the prediction plot of one of my model in this pic


<img width="400" alt="image" src="https://user-images.githubusercontent.com/98169720/150673476-6f2a9d68-983c-4c59-b497-597a8bae703a.png">
<br>

##### see the config of this model in <a href = 'drive.google.com'>this link</a> and <a href = 'drive.google.com'>this link</a>

