import pickle
import pandas as pd

#Run this file only if the above and below vectors have not been created before
#Read the data file (it has been converted to a pickle file)
    pickle_in = open('full_dataset_anon.pickle', "rb")
    data = pickle.load(pickle_in)

#Rename a few columns and convert it to a datetime series
    data = data.rename(columns={'substring': 'patient_id'})
    data['vital_time'] = data['vital_time'].str.split('+').str[0]
    data['vital_time'] = pd.to_datetime(data['vital_time'])

#filter the data that is needed
    vitals = data.filter(['patient_id', 'vital_time','pulse','spo2','resp', 'bpSys', 'temperature'], axis = 1)
    vitals_full_set = vitals[(vitals != 0).all(1)]

#Creating MEWS column
    vitals_full_set = calculate_MEWS(vitals_full_set)

#Removing sensor error
    vitals_clean = unique_chronic_patient(data, chronic_value)

#Saving the clean vitals
    save_as_pickle(vitals_clean, "full_clean_dataset.pickle")

#Create the above and below seven vectors
    vitals_above_seven = []
    vitals_below_seven = []
    for i in vitals_clean['patient_id'].unique():
        this_data = vitals_clean.loc[vitals_clean['patient_id'] == i]
        if this_data['MEWS_clean'].max() >= 7:
            vitals_above_seven.append(this_data)
        else:
            vitals_below_seven.append(this_data)

    vitals_above_seven = pd.concat(vitals_above_seven)
    vitals_below_seven = pd.concat(vitals_below_seven)

    save_as_pickle(vitals_clean, "vitals_above_seven.pickle")
    save_as_pickle(vitals_clean, "vitals_below_seven.pickle")
