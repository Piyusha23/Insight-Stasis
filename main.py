import pandas as pd
from Cleaning_Creating_MEWS import*
from Create_test_train import*
from Model import*



def main():
    print("python main function")

    while True:
        model = input ("Enter the model that you want to run (a) LogRes, (b) SVM, (c) Uni_CNN, (d) Multi_CNN: ")
    if data not in ('LogRes', 'SVM', 'Uni_CNN', 'Multi_CNN'):
            print("Sorry, your entered value was not correct. Make sure the capitalization is correct")
        continue
    else:
            #we're happy with the value given.
            #we're ready to exit the loop.
            break

    pickle_in = open('training_data.pickle', "rb")
    training_data = pickle.load(pickle_in)

    if model == 'LogRes':

        X_train, X_test, y_train, y_test = create_test_train_split(training_data)

        fpr, tpr = Model_logRes(X_train, X_test, y_train, y_test, 'lbfgs')

    elif model == 'SVM':

        X_train, X_test, y_train, y_test = create_test_train_split(training_data)

        fpr_SVM, tpr_SVM = Model_SVM(X_train, X_test, y_train, y_test, 'rbf')

    elif model == 'Uni_CNN':

        X_total, Y_Total = binary_labels_values_univariate(vitals_filter_cont_subset, 7, 3)

        X_train, Y_train, X_test, Y_test = create_test_train_mutli_CNN(X_total, Y_total)

        model_ucnn = Model_Uni_1D_CNN_define(X_train), X_test), n_steps)

        train_acc, test_acc = run_uni_1D_CNN(model, X_train, Y_train, X_test, Y_test, filename)

    else:

        X_Total, Y_Total = binary_labels_values_multivariate(vitals_filter_cont_subset, 7, 5)

        X_train, Y_train, X_test, Y_test = create_test_train_CNN(X_Total, Y_Total)

        model_mcnn = Model_Mutlivariate_1D_CNN_define(X_train_arr, X_test_arr)

        train_acc, test_acc = run_multi_1D_CNN(model, X_train, Y_train, X_test, Y_test, filename)




if __name__ == '__main__':
    main()
