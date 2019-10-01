import pandas as pd
from Cleaning_Creating_MEWS import*
from Create_test_train import*
from Model import*



def main():
    print("python main function")

    pickle_in = open('training_data.pickle', "rb")
    training_data = pickle.load(pickle_in)

    X_train, X_test, y_train, y_test = create_test_train_split(training_data)

    fpr, tpr = Model_logRes(X_train, X_test, y_train, y_test, 'lbfgs')

    fpr_SVM, tpr_SVM = Model_SVM(X_train, X_test, y_train, y_test, 'rbf')

    

if __name__ == '__main__':
    main()
