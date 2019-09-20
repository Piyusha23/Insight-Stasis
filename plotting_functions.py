temp = vitals_train_subset_above.sort_values('year')

fig = plt.figure()
for yy in np.unique(temp['year'].values):
    plt.bar (yy , len(temp.loc[temp['year']== yy]['patient_id'].unique()), label = yy)

plt.legend(loc = 'upper left')
plt.xlabel('Year')
plt.ylabel('Number of unique patients')

def plot_ROC(fpr, tpr):
    from sklearn.metrics import roc_curve, auc

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='SVM')
    plt.plot(fpr_l, tpr_l, color='red', lw=1, label='Logistic Regression')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
