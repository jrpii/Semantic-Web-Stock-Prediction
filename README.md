# Semantic Web Stock Prediction
CSE 573, Group 15, Project 6: Directional Prediction of Stocks

All stock ticker CSV data in OHLCV format. (Date, Time, Open, High, Low, Close, Volume).
	Seperate csv's for 5Min, 15Min, 30Min, 1Hour, 4Hours, 1Day.
	Times are in GMT.

Need to standardize news article times to GMT.


Env Setup
`conda create -n namehere python=3.10
`conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
`pip install -r requirements.txt