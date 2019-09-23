from Cleaning_Creating_MEWS import*
from Create_test_train import*
from Model import*


def main():
    print("python main function")

vitals_above_seven = pd.read_csv('vitals_above_seven.csv')
vitals_below_seven = pd.read_csv('vitals_below_seven.csv')

vitals_above_seven_wanted = filter_wanted_data(vitals_above_seven)
#vitals_below_seven_wanted = filter_wanted_data(vitals_below_seven)

vitals_above_seven_wanted = create_datetime(vitals_above_seven_wanted)
#vitals_above_seven_wanted = create_datetime(vitals_below_seven_wanted)

vitals_above_seven_5min = group_by_minute(vitals_above_seven_wanted, '5Min')
#vitals_above_seven_5min = group_by_minute(vitals_above_seven_wanted, '5Min')

vitals_above_seven_5min = create_feature_vectors(vitals_above_seven_5min, 7, 5)

subset_above, subset_below = get_features_above_and_below_threshold(vitals_above_seven_5min)

vitals_above_seven_subset = get_last_n_points(subset_above, 3)
vitals_below_seven_subset = get_last_n_points(subset_below, 3)

training_data = create_training_data(vitals_below_seven_subset, vitals_above_seven_subset)

save_to_csv(pd.DataFrame(training_data), 'all_labeled_script.csv')

X_train, X_test, y_train, y_test = create_test_train_split(training_data)

fpr, tpr = Model_logRes(X_train, X_test, y_train, y_test, 'lbfgs')

if __name__ == '__main__':
    main()
