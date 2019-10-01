
def get_random_data_sample(data, number):
    patients_to_show = data.sample(number).patient_id.values
    df_sample = data[data.patient_id.isin(patients_to_show)]
    return df_sample

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

def plot_npatients_year(vitals_train_subset_above):

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

def plot_ROC(fpr, tpr, fpr_SVM, tpr_SVM):
    from sklearn.metrics import roc_curve, auc

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='SVM')
    plt.plot(fpr_SVM, tpr_SVM, color='red', lw=1, label='Logistic Regression')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
