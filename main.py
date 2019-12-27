import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense



fileName= 'Jobs.csv';
filePath= 'Dataset' + '/' + fileName;

# -------------------- Tuning Parameters ----------------------

# -------Network Parameters---------------
noOfLSTMUnits= 4;
timeSteps= 2;
outputColumns= [4,5]


# -------Training Parameters---------------
epochs=2
batch_size=10000

# -------------------- Tuning Parameters ----------------------
def customizeDataset(dataset):
	'''Customizing Dataset acc to usage 
	For the Testing Scenario
	Not to be used in production

	Args : 
		dataset : Pandas Dataset
	Returns
		numpyDataset: dataset with selected columns in numpy format
	'''

	filteredDataset= dataset[['SubmitTime', 'WaitTime', 'RunTime', 'NProc', 'UsedCPUTime', 'UsedMemory', 'UserID']]
	# print(filteredDataset.columns)
	print('Orignal Dataset Sample');
	print(filteredDataset.head())

	encodedDataset= pd.get_dummies(filteredDataset, columns=['UserID'], prefix=['UserID_']);
	print('Customized Dataset Sample');
	print(encodedDataset.head())

	numpyDataset= encodedDataset.to_numpy();
	print('Final Dataset(numpy) Shape', numpyDataset.shape)
	return numpyDataset;

def getDataset(filePath):
	'''Get dataset in numpy format
	Args
		filePath: complete filePath
	Returns
		numpyDataset: numpy Dataset 	
	'''
	dataset=pd.read_csv(filePath);
	numpyDataset= customizeDataset(dataset);
	# In case of No Customization
	# numpyDataset= dataset.to_numpy();
	return numpyDataset;



def CreateSequenceDataset(dataset, outputColumns=[ -1] , timeSteps= 1):
	'''
		Create a time based dataset for RNN usage
		Converting [1 2 3 4]T --> [[1 2 3], [2 3 4], [3 4 5] .. ]
	Args:
		data: numpy Dataset 2D dims= [a,b] 
		timeStamp: number of timestamps to cater
	Returns:
		datasetX: numpy 3D Array  dims= [timeSteps, [1,data.shape[1]] ] 
 		datasetY: numpy 3D Array  dims= [data.shape[0], len(outputColumns) ]
	'''
	datasetX= [];
	datasetY= []
	for i in range(0, dataset.shape[0]- timeSteps-1):
		x= dataset[i: i+timeSteps , :]; 
		# Change for exp
		# x= dataset[i: i+timeSteps , 0];
		# print('X : ',x);
		# print('X shape: ', x.shape);
		datasetX.append(x);
		y= dataset[i+timeSteps, outputColumns];
		# print('Y: ',y);
		# print('Y shape: ',y.shape)
		# print(x, ' --> ',y)
		datasetY.append(y)
	datasetX= np.array(datasetX)
	datasetY= np.array(datasetY)
	# print('datasetX shape', datasetX.shape)	
	# print('datasetY shape', datasetY.shape)	
	
	return datasetX,datasetY;

def scaleDataset(dataset, outputColumns):
	'''
	Returns scaled dataset with minMaxScaler in range(0,1)
	along with dataX,DataY scaler
	Args:
		dataset= 2D numpy Array
		outputColumns: columns in dataset used as output
	Returns:
		scaledDataset: 2D scaled Dataset 
		datasetScaler: sklearn.preprocessing.minmaxScaler (X-Scaler)
		outputScaler: sklearn.preprocessing.minmaxScaler (Y-Scaler)	
	'''
	datasetScaler= MinMaxScaler(feature_range=(0,1));
	scaledDataset= datasetScaler.fit_transform(dataset);
	# print('Dataset Scaler Mean: ',datasetScaler.min_);
	# print('Dataset Scaler Scale: ',datasetScaler.scale_);
	
	outputScaler= MinMaxScaler(feature_range=(0,1));
	outputScaler.min_= datasetScaler.min_[outputColumns];
	outputScaler.scale_= datasetScaler.scale_[outputColumns];
	# print('Output Scaler Mean: ',outputScaler.min_);
	# print('Output Scaler Scale: ',outputScaler.scale_);
	# print('Output: ', outputScaler.inverse_transform(scaledDataset[:,outputColumns]))
	return scaledDataset, datasetScaler, outputScaler


def trainTestSplit(dataset,trainSplitPercentage=0.70):
	'''
	Converts Dataset into trains test split 
	Args:
		dataset: numpy 2D dataset
		trainSplitPercentage: (0,1) training data percentage
	Returns:
		train: 2D numpy training Split
		test:  2D numpy testing Split		 
	'''
	# split into train and test sets
	print('Dataset Length: ', len(dataset));
	train_size = int(len(dataset) * trainSplitPercentage)
	test_size = len(dataset) - train_size
	print('Train Size: ', train_size);
	print('Test Size: ', test_size);
	train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
	print(train.shape, test.shape)
	return train,test;

if __name__ == '__main__':
# Dataset Preprocessing 
	orignalDataset= getDataset(filePath);
	dataset, XScaler, YScaler= scaleDataset(orignalDataset, outputColumns)

	train, test= trainTestSplit(dataset,trainSplitPercentage=0.7)
	trainX,trainY= CreateSequenceDataset(train, outputColumns= outputColumns, timeSteps=timeSteps)
	testX,testY= CreateSequenceDataset(test, outputColumns= outputColumns, timeSteps=timeSteps);

# Model Creation and Training.
	
	noOfFeatures= dataset.shape[1];
	model = Sequential();
	model.add(LSTM(noOfLSTMUnits, input_shape=(timeSteps, noOfFeatures)));
	model.add(Dense(2));
	model.compile(optimizer= 'adam', loss='mean_squared_error')
	model.fit(trainX, trainY, epochs=epochs,batch_size=batch_size, verbose=2);

	trianPredict= model.predict(trainX);
	testPredict= model.predict(testX);

# Rescaling to orignal Values
	trianPredict= YScaler.inverse_transform(trianPredict);
	testPredict= YScaler.inverse_transform(testPredict);

	testY= YScaler.inverse_transform(testY);

	# plot baseline and predictions
	plt.plot(testY[0:500,1], label='Orignal Data')
	plt.plot(testPredict[0:500,1], label='Predictions')
	# plt.plot(testPredictPlot)
	plt.show()	
