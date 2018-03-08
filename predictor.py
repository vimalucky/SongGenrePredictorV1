import os
import timeit
import numpy as np
import sys
from collections import defaultdict
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.cross_validation import ShuffleSplit
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from config import TRAIN_DATASET_DIR,TEST_DATASET_DIR,GENRE_LIST
from displayutils import plot_confusion_matrix, plot_roc_curves
from cepsutils import read_train_ceps, read_test_ceps

genre_list = GENRE_LIST

def train_model(X_train, y_train,X_test,y_test, name, plot=False):
    """
        train_model(vector, vector, name[, plot=False])
        
        Trains and saves model to disk.
    """
    labels = np.unique(y_train)

    
    train_errors = []
    test_errors = []

    scores = []
    pr_scores = defaultdict(list)
    precisions, recalls, thresholds = defaultdict(list), defaultdict(list), defaultdict(list)

    roc_scores = defaultdict(list)
    tprs = defaultdict(list)
    fprs = defaultdict(list)

    clfs = []  # for the median

    cms = []

#    print "X_train::"
#    print X_train
#    print "X_test::"
#    print X_test
#    print "y_train::"
#    print y_train
#    print "y_test::"
#    print y_test
    
    clf = LogisticRegression()
    #clf=GaussianNB()
    #clf=SVC(probability=True)
    clf.fit(X_train, y_train)
    clfs.append(clf)

    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print "train_score:: "+str(train_score)
    print "test_score:: "+str(test_score)
    scores.append(test_score)
    
    train_errors.append(1 - train_score)
    test_errors.append(1 - test_score)

    y_pred = clf.predict(X_test)
    print y_pred
    cm = confusion_matrix(y_test, y_pred)
    cms.append(cm)
#    cms = np.asarray(cms)
#    cm_avg = np.mean(cms, axis=0)
#    cm_norm = cm_avg / np.sum(cm_avg, axis=0)
#    plot_confusion_matrix(cm_norm, genre_list, "ceps","CEPS classifier - Confusion matrix")
    

    for label in labels:
        #print "label "+str(label)
        y_label_test = np.asarray(y_test == label, dtype=int)
        #print "y_label_test "+str(y_label_test)
        proba = clf.predict_proba(X_test)
        #print str(len(proba))+"proba "+str(proba)
        proba_label = proba[:, label]
        
        fpr, tpr, roc_thresholds = roc_curve(y_label_test, proba_label)
        roc_scores[label].append(auc(fpr, tpr))
        tprs[label].append(tpr)
        fprs[label].append(fpr)
        
    #sys.exit(1)
    if plot:
        for label in labels:
            scores_to_sort = roc_scores[label]
            median = np.argsort(scores_to_sort)[len(scores_to_sort) / 2]
            desc = "%s %s" % (name, genre_list[label])
            plot_roc_curves(roc_scores[label][median], desc, tprs[label][median],fprs[label][median], label='%s vs rest' % genre_list[label])

    all_pr_scores = np.asarray(pr_scores.values()).flatten()
    summary = (np.mean(scores), np.std(scores), np.mean(all_pr_scores), np.std(all_pr_scores))
    #print("%.3f\t%.3f\t%.3f\t%.3f\t" % summary)

    #save the trained model to disk
    joblib.dump(clf, 'saved_model/model_ceps.pkl')
    
    return np.mean(train_errors), np.mean(test_errors), np.asarray(cms)


if __name__ == "__main__":
    start = timeit.default_timer()
    print " Starting classification \n"
    print " Classification running ... \n" 
    X_train, y_train = read_train_ceps(genre_list)
    
    X_test, y_test = read_test_ceps(genre_list)
    print(len(X_train))
    print(len(X_test))
    
    
    train_avg, test_avg, cms = train_model(X_train, y_train,X_test,y_test, "ceps", plot=True)
    
    cm_avg = np.mean(cms, axis=0)
    cm_norm = cm_avg / np.sum(cm_avg, axis=0)
    print " Classification finished \n"
    stop = timeit.default_timer()
    print " Total time taken (s) = ", (stop - start)
    print "\n Plotting confusion matrix ... \n"
    plot_confusion_matrix(cm_norm, genre_list, "ceps","CEPS classifier - Confusion matrix")
    print " All Done\n"
    print " See plots in 'graphs' directory \n"
    
