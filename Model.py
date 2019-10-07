from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import itertools
import math
import pandas as pd
import matplotlib.pyplot as plt

def create_test_train_split(data):

    X = data[:,:-1]
    Y = data[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test

def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def Model_ARIMA(data_one_patient, p, d, q):
    #The input data_one_patient is the continuous MEWS score for one patient

    #creating test train split - taking 70% of the patient time series as training
    train_len = math.floor(len(data_one_patient)*0.7)
    y_train = data_one_patient['MEWS'].iloc[:train_len]
    y_test = data_one_patient['MEWS'].iloc[train_len:]
    x_train = range(0,len(y_train))

    #parameters for ARIMA
    p = p # Lag order
    d = d # Degree of differencing
    q = q # Moving average order

    ARIMA_model = ARIMA(y_train, order=(p,d,q))
    model_fit = ARIMA_model.fit(disp=0)
    print(model_fit.summary())

    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    plt.show()
    residuals.plot(kind='kde')
    plt.show()
    print(residuals.describe())

    history = y_train.tolist()
    predictions = list()

    # This refits the ARIMA model after every timestep, which is a bit crude and inefficent but it works for now...
    for t in range(len(y_test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = y_test.tolist()[t]
        history.append(obs)

    # Calcualtes mean squared error
    error = mean_squared_error(y_test.tolist(), predictions)
    #print(predictions)
    print('Test MSE: %.3f' % error)

    # Plots test set vs. predictions
    plt.plot(y_test.tolist())
    plt.plot(predictions, color='red')
    plt.grid(True)

#This function runs the model and prints the accuracy. It also returns the fpr, tpr
# to plot the ROC curve for the model.
def Model_logRes(X_train, X_test, y_train, y_test, l_solver):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn import metrics
    from sklearn.metrics import roc_curve, auc
    clf = LogisticRegression(random_state=0, solver=l_solver).fit(X_train, y_train)
    pred = clf.predict(X_test)
    print ('Testing score of LogisticRegression = ', accuracy_score(y_test, pred))
    print ('Training score of LogisticRegression = ', clf.score(X_train, y_train))
    y_score = clf.fit(X_train, y_train).decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    return fpr, tpr

#This function runs the model and prints the accuracy. It also returns the fpr, tpr
# to plot the ROC curve for the model.
def Model_SVM(X_train, X_test, y_train, y_test, kernel_type):
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    from sklearn import metrics
    from sklearn.metrics import roc_curve, auc
    clf_svm = SVC(gamma='auto', kernel=kernel_type)
    clf_svm.fit(X_train, y_train)
    pred = clf_svm.predict(X_test)
    print ('Testing score of SVM = ', accuracy_score(y_test, pred))
    print ('Training score of SVM = ', clf_svm.score(X_train, y_train))
    y_score = clf_svm.fit(X_train, y_train).decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    return fpr, tpr

#Defining the architecture of the 1D univariate CNN model
def Model_Uni_1D_CNN_define(n_steps, n_features):
    from numpy import array
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Flatten
    from keras.layers.convolutional import Conv1D
    from keras.layers.convolutional import MaxPooling1D
    from keras.callbacks import EarlyStopping
    from matplotlib import pyplot
    from keras import regularizers
    from keras.layers import Dropout
    from keras.callbacks import ModelCheckpoint

    #n_steps is the number of training features steps. if n = 3, you have 15 mintes of training data because
    # every step is 5 minutes
    n_steps = n_steps
    n_features = n_features
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features), kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(150, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    model.summary()

    return model

#Compiling and training the univariate 1D CNN model
def run_uni_1D_CNN(model, X_train, Y_train, X_test, Y_test, filename):
    from keras.callbacks import EarlyStopping
    from matplotlib import pyplot
    from keras.callbacks import ModelCheckpoint

    n_features = 1
    X_train_nn = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    X_test_nn = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
    Y_train_nn = Y_train.reshape(-1)
    Y_test_nn = Y_test.reshape(-1)

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    filepath= filename
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # Fit the model
    history = model.fit(X_train_nn, Y_train_nn, validation_split=0.33, epochs=150, batch_size=10, callbacks=callbacks_list, verbose=1)

    tt, train_acc = model.evaluate(X_train_nn, Y_train_nn, verbose=0)
    te, test_acc = model.evaluate(X_test_nn, Y_test_nn, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    # plot training history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='validation')
    pyplot.legend(fontsize = 'large')
    pyplot.xlabel('Epoch', fontsize = 16)
    pyplot.ylabel('Loss', fontsize = 16)
    pyplot.show()

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy', fontsize = 16)
    plt.xlabel('Epoch', fontsize = 16)
    plt.legend(['Train', 'Validation'], loc='upper left', fontsize = 'large')
    plt.show()

    return train_acc, test_acc

#Defining the architecture of the multivariate 1D CNN model
def Model_Mutlivariate_1D_CNN_define(X_train_arr, X_test_arr):

    from numpy import array
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Flatten
    from keras.layers.convolutional import Conv1D
    from keras.layers.convolutional import MaxPooling1D
    from keras import regularizers
    from keras.layers import Dropout
    from keras.callbacks import ModelCheckpoint

    n_steps = X_train_arr.shape[1]
    n_features = X_train_arr.shape[2]
    #num_classes = 2

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features), kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    return model

def run_multi_1D_CNN(model, X_train, Y_train, X_test, Y_test, filename):
    from keras.callbacks import EarlyStopping
    from matplotlib import pyplot



    adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    filepath="weights_1DCNN_nstep10_thresh7.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # Fit the model
    history = model.fit(X_train, Y_train, validation_split=0.33, epochs=150, batch_size=10, callbacks=callbacks_list, verbose=1)

    _, train_acc = model.evaluate(X_train, Y_train, verbose=0)
    _, test_acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    # plot training history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='validation')
    pyplot.legend(fontsize = 'large')
    pyplot.xlabel('Epoch', fontsize = 16)
    pyplot.ylabel('Loss', fontsize = 16)
    pyplot.show()

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy', fontsize = 16)
    plt.xlabel('Epoch', fontsize = 16)
    plt.legend(['Train', 'Validation'], loc='upper left', fontsize = 'large')
    plt.show()

    return train_acc, test_acc

#This function loads the trained 1D CNN model and predicts the labels for your test set
def load_trained_model_uni(filename, X_test, Y_test, n_steps, n_features):
    from numpy import array
    from matplotlib import pyplot
    from tensorflow import keras
    from sklearn.metrics import roc_curve, auc

    model = Model_Uni_1D_CNN_define(n_steps, n_features)
    loaded_model = keras.models.load_model(filename)
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    X_test_nn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    Y_test_nn = Y_test.reshape(-1)

    y_pred_keras = loaded_model.predict(X_test_nn).ravel()

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_test_nn, y_pred_keras)
    auc_keras = auc(fpr_keras, tpr_keras)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

    return y_pred_keras
