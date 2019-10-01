import pickle
import pandas as pd

pickle_in = open('vitals_above_seven.pickle', "rb")
vitals_above_seven = pickle.load(pickle_in)
pickle_in = open('vitals_below_seven.pickle', "rb")
vitals_below_seven = pickle.load(pickle_in)

vitals_above_seven_wanted = filter_wanted_data(vitals_above_seven)

vitals_above_seven_wanted = create_datetime(vitals_above_seven_wanted)

vitals_above_seven_5min = group_by_minute(vitals_above_seven_wanted, '5Min')

vitals_above_seven_5min = create_feature_vectors(vitals_above_seven_5min, 7, 5)

subset_above, subset_below = get_features_above_and_below_threshold(vitals_above_seven_5min 7)

vitals_above_seven_subset = get_last_n_points(subset_above, 3)
vitals_below_seven_subset = get_last_n_points(subset_below, 3)

training_data = create_training_data(vitals_below_seven_subset, vitals_above_seven_subset)

save_as_pickle(training_data, "training_data.pickle")
