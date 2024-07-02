from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
 


class Evaluate:

    def trainAndPredict(X_train, X_test, y_train, y_test):

        # Initialize the SVM classifier (SVC) with an rbf kernel
        clfi = svm.SVC(kernel='rbf', C = 10, gamma = 'auto')

        # Train the SVM classifier using the training data
        clfi.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = clfi.predict(X_test)

        # Evaluate the classifier
        accuracy = clfi.score(X_test, y_test)
        print("Accuracy:", accuracy)
        preds = clfi.predict(X_test)
        labels = y_test

        return preds, labels
    
    def trainAndPredict_LogisticRegression(X_train, X_test, y_train, y_test):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = clf.score(X_test, y_test)
        print("Logistic Regression Accuracy:", accuracy)
        return y_pred, y_test
    
    def trainAndPredict_RandomForest(X_train, X_test, y_train, y_test):
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = clf.score(X_test, y_test)
        print("Random Forest Accuracy:", accuracy)
        return y_pred, y_test
    
    def trainAndPredict_KNeighbors(X_train, X_test, y_train, y_test):
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = clf.score(X_test, y_test)
        print("K-Neighbors Accuracy:", accuracy)
        return y_pred, y_test
    
    def trainAndPredict_DecisionTree(X_train, X_test, y_train, y_test):
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = clf.score(X_test, y_test)
        print("Decision Tree Accuracy:", accuracy)
        return y_pred, y_test
    
    def trainAndPredict_GradientBoosting(X_train, X_test, y_train, y_test):
        clf = GradientBoostingClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = clf.score(X_test, y_test)
        print("Gradient Boosting Accuracy:", accuracy)
        return y_pred, y_test
    

    def calculate_tpr_fpr(self, y_real, y_pred):
        '''
        Calculates the True Positive Rate (tpr) and the True Negative Rate (fpr) based on real and predicted observations

        Args:
            y_real: The list or series with the real classes
            y_pred: The list or series with the predicted classes

        Returns:
            tpr: The True Positive Rate of the classifier
            fpr: The False Positive Rate of the classifier
        '''

        # Calculates the confusion matrix and recover each element
        cm = confusion_matrix(y_real, y_pred)
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]
        TP = cm[1, 1]

        # Calculates tpr and fpr
        tpr =  TP/(TP + FN) # sensitivity - true positive rate
        fpr = 1 - TN/(TN+FP) # 1-specificity - false positive rate

        return tpr, fpr
    

    def get_all_roc_coordinates(self, y_real, y_proba):
        '''
        Calculates all the ROC Curve coordinates (tpr and fpr) by considering each point as a threshold for the predicion of the class.

        Args:
            y_real: The list or series with the real classes.
            y_proba: The array with the probabilities for each class, obtained by using the `.predict_proba()` method.

        Returns:
            tpr_list: The list of TPRs representing each threshold.
            fpr_list: The list of FPRs representing each threshold.
        '''
        tpr_list = [0]
        fpr_list = [0]
        for i in range(len(y_proba)):
            threshold = y_proba[i]
            y_pred = y_proba >= threshold
            tpr, fpr = self.calculate_tpr_fpr(y_real, y_pred)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        return tpr_list, fpr_list
    

    def getResults(preds, labels, num_classes, results, name):

        y_pred = preds
        # Compute accuracy

        accuracy = accuracy_score(labels, y_pred)
        cm = confusion_matrix(labels, y_pred)
        # We will store the results in a dictionary for easy access later
        per_class_accuracies = {}

        # Calculate the accuracy for each one of our classes
        for cls in range(0, num_classes):
            # True negatives are all the samples that are not our current GT class (not the current row)
            # and were not predicted as the current class (not the current column)
            true_negatives = np.sum(np.delete(np.delete(cm, cls, axis=0), cls, axis=1))

            # True positives are all the samples of our current GT class that were predicted as such
            true_positives = cm[cls, cls]

            # The accuracy for the current class is the ratio between correct predictions to all predictions
            per_class_accuracies[cls] = (true_positives + true_negatives) / np.sum(cm)

        #cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Compute precision, recall, F1-score, support for each class
        #acc = cm.diagonal()
        precision, recall, f1, support = precision_recall_fscore_support(labels, y_pred, average=None)
        # Compute micro-average F1-score and ROC AUC

        f1_micro_average = f1_score(labels, y_pred, average='micro')
        print(accuracy, precision, recall, f1, support, f1_micro_average)

        row = {"Name": name, "accuracy":accuracy, "precision": '', "recall": '', "f1":'', "support":'', "f1_micro_average": f1_micro_average}
        new_df = pd.DataFrame([row])
        results = pd.concat([results, new_df])

        for i in range(1, num_classes+1):

            row = {"Name": "Class" + str(i), "accuracy": per_class_accuracies[i-1], "precision": precision[i-1], "recall": recall[i-1], "f1":f1[i-1], "support":support[i-1], "f1_micro_average":''}
            new_df = pd.DataFrame([row])
            results = pd.concat([results, new_df])

        return results
