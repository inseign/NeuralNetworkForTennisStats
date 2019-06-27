# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('womensdataset.csv')
#dataset = dataset.drop(['IndexID'], axis=1)
#dataset = dataset.drop(['TournamentDate'], axis=1)
#dataset = dataset.drop(['Tournament'], axis=1)
#dataset = dataset.drop(['Player1_ID'], axis=1)
#dataset = dataset.drop(['Player2_ID'], axis=1)
#dataset = dataset.drop(['Player1_Hand'], axis=1)
#dataset = dataset.drop(['Player2_Hand'], axis=1)
X = dataset.iloc[:, 0:28].values
y = dataset.iloc[:, 28].values
z = pd.read_csv('women_final_test.csv')
#z = z.drop(['IndexID'], axis=1)
#z = z.drop(['TournamentDate'], axis=1)
#z = z.drop(['Tournament'], axis=1)
z = z.drop(['player_1'], axis=1)
z = z.drop(['player_2'], axis=1)
#z = z.drop(['Player1_Hand'], axis=1)
#z = z.drop(['Player2_Hand'], axis=1)
zx_test = z.iloc[:, 0:28].values
zy_test = z.iloc[:, 28].values
# Encoding categorical data
#from sklearn.preprocessing import OneHotEncoder, LabelEncoder

#encoding data
#labelencoder_X_1 = LabelEncoder()
#X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])
#labelencoder_X_2 = LabelEncoder()
#X[:, 1] = labelencoder_X_2.fit_transform(X[:, 1])
#labelencoder_X_3 = LabelEncoder()
#X[:, 5] = labelencoder_X_3.fit_transform(X[:, 5])
#labelencoder_X_4 = LabelEncoder()
#X[:, 20] = labelencoder_X_4.fit_transform(X[:, 20])

#encoding test
#labelencoder_zx_test_1 = LabelEncoder()
#zx_test[:, 0] = labelencoder_zx_test_1.fit_transform(zx_test[:, 0])
#labelencoder_zx_test_2 = LabelEncoder()
#zx_test[:, 1] = labelencoder_zx_test_2.fit_transform(zx_test[:, 1])
#labelencoder_zx_test_2 = LabelEncoder()
#zx_test[:, 5] = labelencoder_zx_test_2.fit_transform(zx_test[:, 5])
#labelencoder_zx_test_2 = LabelEncoder()
#zx_test[:, 20] = labelencoder_zx_test_2.fit_transform(zx_test[:, 20])

#removing dummy variables
#onehotencoder = OneHotEncoder(categorical_features = [0])
#X = onehotencoder.fit_transform(X).toarray()
#X = X[:, 1:]
#zx_test = onehotencoder.fit_transform(zx_test).toarray()
#zx_test = zx_test[:, 1:]
#onehotencoder = OneHotEncoder(categorical_features = [2])
#X = onehotencoder.fit_transform(X).toarray()
#X = X[:, 1:]
#zx_test = onehotencoder.fit_transform(zx_test).toarray()
#zx_test = zx_test[:, 1:]
#onehotencoder = OneHotEncoder(categorical_features = [12])
#X = onehotencoder.fit_transform(X).toarray()
#X = X[:, 1:]
#zx_test = onehotencoder.fit_transform(zx_test).toarray()
#zx_test = zx_test[:, 1:]
#onehotencoder = OneHotEncoder(categorical_features = [29])
#X = onehotencoder.fit_transform(X).toarray()
#X = X[:, 1:]
#zx_test = onehotencoder.fit_transform(zx_test).toarray()
#zx_test = zx_test[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 63)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
zx_test = sc.transform(zx_test)
# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu', input_dim = 28))

# Adding the second hidden layer
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
i=0
while i < 5:
    classifier.fit(X_train, y_train, batch_size = 1, epochs = 1)

    # Part 3 - Making the predictions and evaluating the model

    # Predicting the Test set results
    y_pred_odds1 = classifier.predict(X_test)
    y_pred_odds = classifier.predict(zx_test)
    y_pred = (y_pred_odds1 > 0.5)
    y_pred2 = (y_pred_odds > 0.5)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    #cm2 = confusion_matrix(zy_test, y_pred2)
    Accu1=(cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[1,0]+cm[0,1])
    print(Accu1)
    #Accu2=(cm2[0,0]+cm2[1,1])/(cm2[0,0]+cm2[1,1]+cm2[1,0]+cm2[0,1])
    #print(Accu2)
    i+=1

#(cm2[0,0]+cm2[1,1])/(cm2[0,0]+cm2[1,1]+cm2[1,0]+cm2[0,1])
#(cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[1,0]+cm[0,1])

#exporting to excel spreadsheet 1485
df = pd.DataFrame (y_pred_odds)
#df2 = pd.DataFrame (X_test)
#df3 = pd.DataFrame (y_pred_odds1)

## save to xlsx file
filepath = 'TestResultsPTA2018.xlsx'
#filepath2 = 'TestX.xlsx'
#filepath3 = 'TestResultsX.xlsx'
df.to_excel(filepath, index=False)
#df2.to_excel(filepath2, index=False)
#df3.to_excel(filepath3, index=False)
