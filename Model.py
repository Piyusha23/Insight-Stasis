from sklearn.model_selection import train_test_split

def create_test_train_split(data):
    X = data[:,0:3]
    Y = data[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test

#This function runs the model and prints the accuracy. It also returns the fpr, tpr
# to plot the ROC curve for the model.
def Model_logRes(X_train, X_test, y_train, y_test, l_solver):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn import metrics
    from sklearn.metrics import roc_curve, auc
    clf = LogisticRegression(random_state=0, solver=l_solver).fit(X_train, y_train)
    pred = clf.predict(X_test)
    print ('Testing score of LogisticRegression = ', accuracy_score(y_test, pred))
    print ('Training score of LogisticRegression = ', clf.score(X_train, y_train))
    y_score = clf.fit(X_train, y_train).decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    return fpr, tpr

#This function runs the model and prints the accuracy. It also returns the fpr, tpr
# to plot the ROC curve for the model.
def Model_SVM(X_train, X_test, y_train, y_test, kernel_type):
    from sklearn.svm import SVC
    clf_svm = SVC(gamma='auto', kernel='kernel_type')
    clf_svm.fit(X_train, y_train)
    pred = clf_svm.predict(X_test)
    print ('Testing score of SVM = ', accuracy_score(y_test, pred))
    print ('Training score of SVM = ', clf.score(X_train, y_train))
    y_score = clf_svm.fit(X_train, y_train).decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    return fpr, tpr
