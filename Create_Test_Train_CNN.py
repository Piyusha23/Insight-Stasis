import pandas as import pd

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

if input == 'Uni':
    subset = vitals_filter_cont_filter.filter(['MEWS_clean','continuous'],axis = 1)
    X_Total_D, Y_Total_D = binary_labels_values_univariate(subset, 7, 3)

vitals_filter_cont_subset = vitals_filter_cont_filter.filter(['bpSys','pulse','resp','spo2','temperature','MEWS_clean','patient_id','continuous'], axis = 1)

X_Total, Y_Total = binary_labels_values_multivariate(vitals_filter_cont_subset, 7, 5)

X_train, Y_train, X_test, Y_test = create_test_train_multi_CNN(X_Total, Y_Total)
