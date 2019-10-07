from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_random_patient_vital(data, vital):
    patients_to_show = data.sample(3).patient_id.values
    df_sample = data[data.patient_id.isin(patients_to_show)]
    fig = plt.figure()
    plot_value = vital
    for i in data_sample['patient_id'].unique():
        this_data = data_sample.loc[data_sample['patient_id'] == i]
        plt.plot(np.arange(len(this_data)), this_data[plot_value].values, label = i)
        plt.legend()
        plt.title('plot_value')
    return fig

#plotting the vital signs for a random sample of patients

def plot_all_vital_signs(dataframe, patient_id):

    fig = plt.figure()

    #for i in vitals_above_seven['patient_id'].unique():
    this_data = vitals_above_seven.loc[vitals_above_seven['patient_id'] == patient_id]
    ['bpSys'].plot()
    plt.plot(this_data['pulse'], label="Pulse")
    plt.plot(this_data['spo2'], label="spO2")
    plt.plot(this_data['temperature'], label="Temp")
    plt.plot(this_data['resp'], label="Resp")
    plt.plot(this_data['bpSys'], label="Sys BP")
    plt.plot(this_data['MEWS_clean'], '-.', label = "MEWS")
    plt.legend(loc='upper left')
    plt.xlabel('Time', fontsize = 16)
    plt.ylabel('Vital Sign', fontsize = 16)
    plt.show()


def plot_MEWS(dataframe):

    fig = plt.figure()
    #for i in vitals_above_seven['patient_id'].unique():
    this_data = dataframe.loc[dataframe['patient_id'] == 'fb87a30bc']
    plt.plot(this_data['MEWS_clean'], label = "MEWS")
    plt.legend(loc = 'upper left')
    plt.xlabel('Time', fontsize = 18)
    plt.ylabel('MEWS', fontsize = 18)

def plot_ROC(fpr, tpr):

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='LogisticRegression')
    #plt.plot(fpr_SVM, tpr_SVM, color='red', lw=1, label='Logistic Regression')
    #plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

def plotting_wrongly_classified(Y_test_arr, X_test_arr, y_pred_test, pred_label):
    import random

    #Reshape arrays
    X_test_nn = X_test_arr.reshape((X_test_arr.shape[0], X_test_arr.shape[1], n_features))
    Y_test_nn = Y_test_arr.reshape(-1)
    Y_test = pd.DataFrame(Y_test_nn)

    y_pred_test[y_pred_test < 0.5] = 0
    y_pred_test[y_pred_test >= 0.5] = 1

    #Get indices and waveforms of misclassified values
    bool_v = (Y_test[0] == y_pred_test)*1
    nn = X_test_nn[bool_v.index[bool_v == 0].tolist()]
    yy = Y_test_nn[bool_v.index[bool_v == 0].tolist()]
    yy = pd.DataFrame(yy)
    yy_misclassified_as_zero = yy[yy[0] == 0]
    yy_misclassifed_as_one = yy[yy[0] == 1]
    nn_misclassified_as_zero = nn[yy_misclassified_as_zero.index].tolist()
    nn_misclassified_as_one = nn[yy_misclassifed_as_one.index].tolist()

    #Plotting misclassified values

    for i in range(1,10):
        plt.figure
        plt.subplot(3,3,i)
        if pred_label == "stable":
            a = random.sample(range(1, len(nn_misclassified_as_zero)), 10)
            plt.plot(nn_misclassified_as_zero[a[i]])
            plt.suptitle('Wrongly predicted as stable (MEWS < 7) - should be unstable')
        else:
            a = random.sample(range(1, len(nn_misclassified_as_one)), 10)
            plt.plot(nn_misclassified_as_one[a[i]])
            plt.suptitle('Wrongly predicted as unstable (MEWS > 7) - should be stable')
