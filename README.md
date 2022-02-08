# Bitcoin price predictor 
predict price of bitcoin with python and tensorflow 
<br><br>

## Get dataset
Get edited dataset from drive link : <a href = 'https://drive.google.com/file/d/1CjZB1DWnM_BMzg-EeE6PfyUeXvhW_TMg/view?usp=sharing'>click hare for google drive link</a><br>
or orginal dataset from kaggle link : <a href='https://www.kaggle.com/tencars/392-crypto-currency-pairs-at-minute-resolution'>click hare for kaggle link</a><br>
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


<img width="400" alt="image" src="https://github.com/BofSbitCode/bitcoin-price-predictor/blob/main/savedModel/prediction%20days%20%3D%2060/date%20%3D%202022-02-06/name%20%3D%20test-1/prediction%20plot%20_%20testSize%20%3D%20365.png">

##### see the config of this model in <a href = 'https://github.com/BofSbitCode/bitcoin-price-predictor/blob/main/savedModel/prediction%20days%20%3D%2060/date%20%3D%202022-02-06/name%20%3D%20test-1/modelSummary.info'>this link</a> and <a href = 'https://github.com/BofSbitCode/bitcoin-price-predictor/blob/main/savedModel/prediction%20days%20%3D%2060/date%20%3D%202022-02-06/name%20%3D%20test-1/info.json'>this link</a>

#### Warning ⛔  : never use this for trade, it's just study project but it can be developed for a trade bot, but it's still untrustable


