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
    features = np.unique(dataframe['patient_id'])
    dataframe['delta'] = (dataframe['datetime']-dataframe['datetime'].shift(1)).astype('timedelta64[m]')

    new_df = pd.DataFrame()
    for pi in tqdm(features):
        group_by_minute = dataframe.loc[dataframe['patient_id'] == pi]
        group_by_minute.index = pd.to_datetime(group_by_minute['datetime'].values)
        group_by_minute = group_by_minute.groupby(pd.TimeGrouper(minute)).median()
        #group_by_minute['patient_id'] = [pi for zzz in range(len(group_by_minute))]
        group_by_minute['patient_id'] = pi

        new_df = new_df.append(group_by_minute)
        new_df = new_df.dropna()
        new_df['delta'] = (new_df['datetime']-new_df['datetime'].shift(1)).astype('timedelta64[m]')

    return new_df

#Creating the feature vector after grouping the variables by the sampling time desired.
#Create a new feature - this represents the engineered feature, i.e. the broken up parts
# of the time series. The time series is broken at the peaks - this can be 7 or greater.
def create_feature_vectors(dataframe, desired_peak_value, sampling_value):
    temp = 0
    keep = []
    for i in tqdm(dataframe['patient_id'].unique()):
        temp = temp+1
        this_data = dataframe.loc[dataframe['patient_id'] == i]
        for j in range(len(this_data)):
            if j == 0:
                keep.append(temp)
            elif (((this_data['MEWS_clean'].iloc[j] < desired_peak_value) & (this_data['MEWS_clean'].iloc[j-1] >= desired_peak_value)) | (this_data['delta'].iloc[j] != sampling_value)):
                temp+=1
                keep.append(temp)
            else:
                keep.append(temp)
    dataframe['feature_new'] = keep
    return dataFrame

#Converting time to datetime and dropping the vital time column
def create_datetime(data, time_index_name):
    data['datetime'] = pd.to_datetime(data['time_index_name'])
    data.drop(['time_index_name'], axis=1, inplace=True)

def remove_small_features(dataframe):
    subset = []
    for i in tqdm(dataframe['feature_new'].unique()):
        this_data = dataframe.loc[new_5min_df['feature_new'] == i]
        if len(this_data) >= 3:
            subset.append(this_data)

    return subset


def get_features_above_and_below_threshold(dataframe):
    subset_above_seven = []
    subset_below_seven = []
    for i in subset['feature_new'].unique():
        this_data = subset.loc[subset['feature_new'] == i]
        if this_data['MEWS_clean'].max() >= 7:
            subset_above_seven.append(this_data)
            #vitals_above_seven = pd.concat(vitals_above_seven)
        else:
            subset_below_seven.append(this_data)

    subset_above_seven = pd.concat(subset_above_seven)
    subset_below_seven = pd.concat(subset_below_seven)

    return subset_above_seven, subset_below_seven

def get_last_n_points(dataframe, n):
    new_frame = []
    for i in dataframe['feature'].unique():
        this_data = dataframe.loc[dataframe['feature'] == i]
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
