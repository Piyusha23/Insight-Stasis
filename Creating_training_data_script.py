import pandas as pd
from Cleaning_Creating_MEWS import*
from Create_test_train import*
from Model import*

#Enter the model you want to process your training data for
#Each model needs data to pre-processed a certain way

while True:
    model = input ("Enter the model that you want to run (a) LogRes, (b) SVM, (c) Uni_CNN, (d) Multi_CNN: ")
    if model not in ('LogRes', 'SVM', 'Uni_CNN', 'Multi_CNN'):
        print("Sorry, your entered value was not correct. Make sure the capitalization is correct")
        continue
    else:
        break



pickle_in = open('vitals_above_seven.pickle', "rb")
vitals_above_seven = pickle.load(pickle_in)
pickle_in = open('vitals_below_seven.pickle', "rb")
vitals_below_seven = pickle.load(pickle_in)

vitals_all = pd.concat([vitals_above_seven,vitals_below_seven], axis = 0)

vitals_filter = create_datetime(vitals_all)

#Minute denotes the desired sampling frequency of your dataset
vitals_filter_5min = group_by_minute(vitals_filter, '5Min')

#Group by the continuous chunks of 5min data that exists in your dataframe
vitals_filter_cont = length_continuous_data(vitals_filter_5min, 5)

#Filter the number of continuous values you want to include in your training/testing data
vitals_filter_cont_filter = filter_continuous_data(vitals_filter_cont, 20)

if model == 'LogRes':

    pickle_in = open('training_data.pickle', "rb")
    training_data = pickle.load(pickle_in)

    X_train, X_test, y_train, y_test = create_test_train_split(training_data)

    fpr, tpr = Model_logRes(X_train, X_test, y_train, y_test, 'lbfgs')

elif model == 'SVM':

    pickle_in = open('training_data.pickle', "rb")
    training_data = pickle.load(pickle_in)

    X_train, X_test, y_train, y_test = create_test_train_split(training_data)

    fpr_SVM, tpr_SVM = Model_SVM(X_train, X_test, y_train, y_test, 'rbf')

elif model == 'Uni_CNN':

    #For univariate CNN the input time series consists only of the MEWS
    subset = vitals_filter_cont_filter.filter(['MEWS_clean','continuous'],axis = 1)

    threshold = 7 #Selecting the threshold you want for defining "bad" outcome
    nsteps = 3 #Defining the length of sliding window for the 1D CNN.Each step corresponds to 5 minutes

    #This function creates the input (feature engineering) for 1D CNN with labels for features.
    X_Total_D, Y_Total_D = binary_labels_values_univariate(subset, threshold, nsteps)

    #Making sure the dataset is balanced and creation of test,train split
    X_train, Y_train, X_test, Y_test = create_test_train_uni_CNN(X_Total_D, Y_Total_D)

    #Saving the training data for use later
    save_as_pickle(X_train, "X_train_Uni.pickle")
    save_as_pickle(Y_train, "Y_train_Uni.pickle")
    save_as_pickle(X_test, "X_test_Uni.pickle")
    save_as_pickle(Y_test, "Y_test_Uni.pickle")

else:

    threshold = 7 #Selecting the threshold you want for defining "bad" outcome
    nsteps = 3 #Defining the length of sliding window for the 1D CNN.Each step corresponds to 5 minutes

    #This function creates the input (feature engineering) for 1D multivariate CNN with labels for features.
    X_Total, Y_Total = binary_labels_values_multivariate(vitals_filter_cont_subset, threshold, nsteps)

    X_train, Y_train, X_test, Y_test = create_test_train_multi_CNN(X_Total, Y_Total)

    #Saving the training data for use later
    save_as_pickle(X_train, "X_train_Multi.pickle")
    save_as_pickle(Y_train, "Y_train_Multi.pickle")
    save_as_pickle(X_test, "X_test_Multi.pickle")
    save_as_pickle(Y_test, "Y_test_Multi.pickle")
