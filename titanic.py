# Importing the libraries
import pandas as pd
import keras

# Importing the dataset and preprocessing it
dataset1 = pd.read_csv('gender_submission.csv')
dataset2 = pd.read_csv('test.csv')
dataset3 = pd.read_csv('train.csv')
from sklearn.preprocessing import Imputer #object to replace missing data with mean 
imputer = Imputer()
dataset_train = dataset3.iloc[:,[1,2,4,5,6,7]] #prepare training set
dataset_train['Age'] = imputer.fit_transform(dataset_train.iloc[:,[3]]) #replace missing data with mean
y_train = dataset_train.iloc[:,0].values             #dependant variable
X_train = dataset_train.iloc[:,1:].values            #independant variables
dataset_test = dataset2.iloc[:,[1,3,4,5,6]]   #prepare test dataset
dataset_test['Age'] = imputer.fit_transform(dataset_test.iloc[:,[2]]) #replace missing data with mean
X_test = dataset_test.iloc[:,0:5].values   #dependant variables
y_test = dataset1.iloc[:,1].values   #independant variables

#process categorical data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder1 = LabelEncoder()  #object to encode categorical data
X_train[:, 0] = labelencoder1.fit_transform(X_train[:, 0])
labelencoder2 = LabelEncoder()
X_train[:, 1] = labelencoder2.fit_transform(X_train[:,1])
labelencoder3 = LabelEncoder()
X_test[:, 0] = labelencoder3.fit_transform(X_test[:, 0])
labelencoder4 = LabelEncoder()
X_test[:, 1] = labelencoder4.fit_transform(X_test[:, 1])
OHT = OneHotEncoder(categorical_features = [0])  #object to represent each feature
X_train = OHT.fit_transform(X_train).toarray()   # with one column in a sparse matrix
X_test = OHT.transform(X_test).toarray()
X_train = X_train[:, 1:]
X_test = X_test[:, 1:]

#Feature Scaling
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
X_train = SS.fit_transform(X_train)
X_test = SS.transform(X_test)

#creat ANN
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

classifier.add(Dense(output_dim = 4, init = 'glorot_uniform', activation = 'relu', input_dim = 6)) #input layer

classifier.add(Dense(output_dim = 4, init = 'glorot_uniform', activation = 'relu')) #1st hidden layer

classifier.add(Dense(output_dim = 4, init = 'glorot_uniform', activation = 'relu')) #2nd hiddeen layer

classifier.add(Dense(output_dim = 1, init = 'glorot_uniform', activation = 'sigmoid')) #output layer

classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy']) #compile ANN

classifier.fit(X_train, y_train, batch_size = 32, nb_epoch = 200) #fit ANN to training set

# Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



