import pandas as pd
import numpy as np
import seaborn as sns
import math
import pickle
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from datetime import datetime

def save_as_pickle(data, filename):
    pickle_out = open(filename, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()

def rename_columns(data, column_to_name, new_name):
    return data.rename(columns={'column_to_name':'new_name'})

def number_of_unique_patients(data, id):
    unique_ids = data[id].unique()
    len_unique_ids = len(data[id].unique())
    print ('Number of patients in dataset = ', len_unique_ids)

def filter_wanted_data(data):
    vitals = data.filter(['patient_id', 'vital_time','pulse','spo2','resp', 'bpSys', 'temperature', 'MEWS_clean'], axis = 1)
    #data = vitals[(vitals != 0).all(1)]
    return data

def calculate_MEWS(data):
    data.loc[(data['resp'] > 11)| (data['resp'] < 21),'MEWS_resp'] = 0
    data.loc[(data['resp'] > 8) & (data['resp'] < 12),'MEWS_resp'] = 1
    data.loc[(data['resp'] > 20) & (data['resp'] < 25),'MEWS_resp'] = 2
    data.loc[(data['resp'] >= 25) | (data['resp'] <= 8),'MEWS_resp'] = 3

    data.loc[(data['pulse'] > 50)& (data['pulse'] < 90),'MEWS_pulse'] = 0
    data.loc[((data['pulse'] > 40) & (data['pulse'] <= 50)) | ((data['pulse'] >= 90) & (data['pulse'] < 110)),'MEWS_pulse'] = 1
    data.loc[(data['pulse'] >= 110) & (data['pulse'] < 130),'MEWS_pulse'] = 2
    data.loc[(data['pulse'] <= 40) | (data['pulse'] >= 130),'MEWS_pulse'] = 3

    data.loc[data['spo2'] >=96,'MEWS_spo2'] = 0
    data.loc[(data['spo2'] >= 94) & (data['spo2'] < 96),'MEWS_spo2'] = 1
    data.loc[(data['spo2'] > 91) & (data['spo2'] < 94),'MEWS_spo2'] = 2
    data.loc[data['spo2'] <= 91,'MEWS_spo2'] = 3

    data.loc[(data['temperature'] > 36)& (data['temperature'] < 38),'MEWS_temperature'] = 0
    data.loc[(data['temperature'] >= 38) & (data['temperature'] < 39) | (data['temperature'] > 35) & (data['temperature'] <= 36),'MEWS_temperature'] = 1
    data.loc[(data['temperature'] >= 39),'MEWS_temperature'] = 2
    data.loc[(data['temperature'] <= 35),'MEWS_temperature'] = 3

    data.loc[(data['bpSys'] > 110)& (data['bpSys'] < 220),'MEWS_bpSys'] = 0
    data.loc[(data['bpSys'] > 100) & (data['bpSys'] <= 110),'MEWS_bpSys'] = 1
    data.loc[(data['bpSys'] > 90) & (data['bpSys'] <= 100),'MEWS_bpSys'] = 2
    data.loc[(data['bpSys'] <= 90) | (data['bpSys'] >= 220),'MEWS_bpSys'] = 3

    data['MEWS'] = data['MEWS_resp']+data['MEWS_temperature']+data['MEWS_spo2']+data['MEWS_pulse']+data['MEWS_bpSys']

    return data

def unique_chronic_patient(data, chronic_value):
#This function also removes sensor error. The unique value (or greater than it) must be
#repeated at least twice.
    data['next_MEWS'] = data.MEWS.shift(-1)
    data['prev_MEWS'] = data.MEWS.shift(1)
    data['MEWS_clean'] = data['MEWS']
    data.loc[(data['MEWS'] >= chronic_value) & (data['next_MEWS'] < chronic_value) & (data['prev_MEWS'] < chronic_value), 'MEWS_clean'] = 100
    vitals_clean = data[data['MEWS_clean'] < 100]
    unique_ids = vitals_clean['patient_id'].unique()
    len_unique_ids = len(vitals_clean['patient_id'].unique())
    print ('Number of patients in dataset with sensor error removed = ', len_unique_ids)

    return data

def save_to_csv(data, name):
    data.to_csv(name)
