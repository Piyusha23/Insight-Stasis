import pandas as pd
import numpy as np
import seaborn as sns
import math
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from datetime import datetime

# There are lots of duplicate values over seconds or there are sampling discrepancies, i.e,
# things are sampled over a couple of minutes. This functions takes the dataframe and the
# desired sampling rate as the input and combines the values of the intermediate sampling values
# by taking the median of the values.

def group_by_minute(dataframe, minute):
    from tqdm import tqdm

    features = dataframe['patient_id'].unique()
    #dataframe['delta'] = (dataframe['datetime']-dataframe['datetime'].shift(1)).astype('timedelta64[m]')

    new_df = pd.DataFrame()
    for pi in tqdm(features):
        group_by_minute = dataframe.loc[dataframe['patient_id'] == pi]
        group_by_minute.index = pd.to_datetime(group_by_minute['datetime'].values)
        group_by_minute = group_by_minute.sort_index()
        group_by_minute = group_by_minute.groupby(pd.TimeGrouper(minute)).median()
        #group_by_minute['patient_id'] = [pi for zzz in range(len(group_by_minute))]
        group_by_minute['patient_id'] = pi

        new_df = new_df.append(group_by_minute)
        new_df = new_df.dropna()

    return new_df


#Creating the feature vector after grouping the variables by the sampling time desired.
#Create a new feature - this represents the engineered feature, i.e. the broken up parts
# of the time series. The time series is broken at the peaks - this can be 7 or greater.
def create_feature_vectors(dataframe, threshold_value, delta_t):
    from tqdm import tqdm
    temp = 0
    keep = []
    dataframe['datetime'] = dataframe.index
    dataframe['delta'] = (dataframe['datetime']-dataframe['datetime'].shift(1)).astype('timedelta64[m]')
    for i in tqdm(dataframe['patient_id'].unique()):
        temp = temp+1
        this_data = dataframe.loc[dataframe['patient_id'] == i]
        for j in range(len(this_data)):
            if j == 0:
                keep.append(temp)
            elif (((this_data['MEWS_clean'].iloc[j] < threshold_value) & (this_data['MEWS_clean'].iloc[j-1] >= threshold_value)) | (this_data['delta'].iloc[j] != delta_t)):
                temp+=1
                keep.append(temp)
            else:
                keep.append(temp)

    print (this_data.shape)
    dataframe['feature_new'] = keep

    dataframe.dropna()

    return dataframe

#Converting time to datetime and dropping the vital time column
def create_datetime(data):
    data['datetime'] = pd.to_datetime(data['vital_time'])
    data.drop(['vital_time'], axis=1, inplace=True)
    return data

def remove_small_features(dataframe):
    subset = []
    for i in tqdm(dataframe['feature_new'].unique()):
        this_data = dataframe.loc[new_5min_df['feature_new'] == i]
        if len(this_data) >= 3:
            subset.append(this_data)

    return subset


def get_features_above_and_below_threshold(subset_dataframe, threshold_value):
    subset_above_seven = []
    subset_below_seven = []
    for i in subset_dataframe['feature_new'].unique():
        this_data = subset_dataframe.loc[subset_dataframe['feature_new'] == i]
        if (this_data['MEWS_clean'].max() >= threshold_value) & (len(this_data) >= 3):
            subset_above_seven.append(this_data)
            #vitals_above_seven = pd.concat(vitals_above_seven)
        else:
            subset_below_seven.append(this_data)

    subset_above_seven = pd.concat(subset_above_seven)
    subset_below_seven = pd.concat(subset_below_seven)

    return subset_above_seven, subset_below_seven

def get_last_n_points(dataframe, n):
    from tqdm import tqdm
    new_frame = []
    for i in tqdm(dataframe['feature_new'].unique()):
        this_data = dataframe.loc[dataframe['feature_new'] == i]
        aa = this_data['MEWS_clean'].values[-n:]
        if len(aa) >= n:
            new_frame.append(aa)

    return new_frame

def create_training_data(data_below_seven, data_above_seven):
    subset_above = np.array(data_above_seven)
    subset_below = np.array(data_below_seven)
    label_ones = np.ones(len(subset_above))
    label_ones = np.array(label_ones)
    label_zeros = np.zeros(len(subset_below))
    label_zeros = np.array(label_zeros)
    training_above_with_labels = np.column_stack((subset_above,label_ones))
    training_below_with_labels = np.column_stack((subset_below,label_zeros))
    all_labeled_data = np.concatenate((training_below_with_labels, training_above_with_labels), axis=0)
    np.random.shuffle(all_labeled_data)

    return all_labeled_data

def length_continuous_data(dataframe, delta_t):
    from tqdm import tqdm
    temp = 0
    keep = []
    dataframe['datetime'] = dataframe.index
    dataframe['delta'] = (dataframe['datetime']-dataframe['datetime'].shift(1)).astype('timedelta64[m]')
    for i in tqdm(dataframe['patient_id'].unique()):
        temp = temp+1
        this_data = dataframe.loc[dataframe['patient_id'] == i]
        for j in range(len(this_data)):
            if j == 0:
                keep.append(temp)
            elif ((this_data['delta'].iloc[j] != delta_t)):
                temp+=1
                keep.append(temp)
            else:
                keep.append(temp)

    print (this_data.shape)
    dataframe['continuous'] = keep

    dataframe.dropna()

    return dataframe

# Format the training/testing data for univariate single step 1D CNN 
def split_sequence_uni(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix-1:]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Format the training/testing data for multivariate 1D CNN
def split_sequences_multivariate(sequences, n_steps):
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
    return X, y

# Format the training/testing data for univariate multi-step 1D CNN
def split_sequences_multistep(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-3:, -1]
        X.append(seq_x)
        y.append(seq_y)
    return X, y


def binary_labels_values_multivariate(df, binary_threshold, n_steps):
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np

    from tqdm import tqdm
    scaler = MinMaxScaler()
    X_Total = list()
    Y_Total = list()
    thresh = binary_threshold #threshold for binary classification
    n_steps = n_steps #number of time steps to look at

    for i in tqdm(df['continuous'].unique()):
        this_data = df.loc[df['continuous'] == i]
        this_data_train = this_data[['bpSys','pulse','resp','spo2','temperature']]
        this_data_labels = this_data['MEWS_clean']
        this_data_labels = this_data_labels.apply(lambda x: 0 if (x < thresh) else 1)
        labels = this_data_labels.reset_index(drop = True)
        scaler.fit(this_data_train)
        this_data_scaled = scaler.transform(this_data_train)
        this_data_values = pd.DataFrame(this_data_scaled)
        this_data_total = pd.concat([this_data_values,labels], axis=1)

        X, Y = split_sequences_multivariate(this_data_total.values, n_steps)

        X_Total.extend(X)
        Y_Total.extend(Y)
    return np.array(X_Total), np.array(Y_Total)

def filter_continuous_data(dataframe, n_cont_values):
    cont_number = 20

    aa = dataframe['continuous'].value_counts()

    aa_df = pd.DataFrame()
    aa_df['continuous_val'] = aa.keys()
    aa_df['continuous_freq'] = aa.values

    aa_df_subset = aa_df[aa_df['continuous_freq'] >= cont_number]

    data_subset = dataframe.loc[dataframe['continuous'].isin(aa_df_subset['continuous_val'].values)]

    return data_subset

def create_test_train_CNN(X_tot, Y_tot):
    from random import shuffle
    total = list(zip(X_tot, Y_tot))
    shuffle(total)

    X_total = [total[i][0] for i in range(len(total))]
    labels = [total[i][1] for i in range(len(total))]

    print ("Shape of X_total = ", len(X_total))
    print ("Shape of labels = ", len(labels))

    X_total = np.asarray(X_total)
    Y_total = np.asarray(labels)

    X_train = X_total[:math.floor(len(X_total)*0.7)]
    Y_train = Y_total[:math.floor(len(Y_total)*0.7)]

    #Is the dataset balanced?
    uniqueValues, occurCount = np.unique(Y_total, return_counts=True)
    print(uniqueValues, occurCount)
    print("Shape of Y_train = ", Y_train.shape)
    print("Shape of X_train = ", X_train.shape)

    X_test = X_total[(math.floor(len(X_total)*0.7)):]
    Y_test = Y_total[(math.floor(len(Y_total)*0.7)):]

    uniqueValues, occurCount = np.unique(Y_test, return_counts=True)
    print(uniqueValues, occurCount)
    print("Shape of Y_train = ", X_test.shape)
    print("Shape of X_train = ", Y_test.shape)
    #Y_test = Y_Total_N[(len(Y_Total_N) - math.floor(len(Y_Total_N)*0.7)):]

    print ("X_train Shape = ", X_train.shape)
    print ("X_test Shape = ", X_test.shape)
    print ("Y_train Shape = ", Y_train.shape)
    print ("Y_test Shape = ", Y_test.shape)

    return X_train, Y_train, X_test, Y_test
