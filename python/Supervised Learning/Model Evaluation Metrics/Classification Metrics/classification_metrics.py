# Import our libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, fbeta_score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy import interp
# import tests as t

# Read in our dataset
df = pd.read_csv('smsspamcollection_SMSSpamCollection',
                 header=None,
                 names=['label', 'sms_message'])

# Fix our response value
df['label'] = df.label.map({'ham': 0, 'spam': 1})

# Split our dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
                                                    df['label'],
                                                    random_state=1)

# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the
# testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)

# Instantiate a number of our models
naive_bayes = MultinomialNB()
bag_mod = BaggingClassifier(n_estimators=200)
rf_mod = RandomForestClassifier(n_estimators=200)
ada_mod = AdaBoostClassifier(n_estimators=300, learning_rate=0.2)
svm_mod = SVC()

# Fit each of the 4 models
# This might take some time to run
naive_bayes.fit(training_data, y_train)
bag_mod.fit(training_data, y_train)
rf_mod.fit(training_data, y_train)
ada_mod.fit(training_data, y_train)
svm_mod.fit(training_data, y_train)


# Make predictions using each of your models
nb = naive_bayes.predict(testing_data)
bag_pred = bag_mod.predict(testing_data)
rf_pred = rf_mod.predict(testing_data)
ada_pred = ada_mod.predict(testing_data)
svm_pred = svm_mod.predict(testing_data)


# accuracy is the total correct divided by the total to predict
def accuracy(actual, preds):
    '''
    INPUT
    preds - predictions as a numpy array or pandas series
    actual - actual values as a numpy array or pandas series

    OUTPUT:
    returns the accuracy as a float
    '''
    return np.sum(preds == actual) / len(actual)


print(accuracy(y_test, nb))
print(accuracy_score(y_test, nb))
print("Since these match, we correctly calculated our metric!")


# precision is the true positives over the predicted positive values
def precision(actual, preds):
    '''
    INPUT
    (assumes positive = 1 and negative = 0)
    preds - predictions as a numpy array or pandas series
    actual - actual values as a numpy array or pandas series

    OUTPUT:
    returns the precision as a float
    '''
    TP = np.sum((preds == actual) & (preds > 0))
    FP = np.sum((preds == 1) & (actual == 0))
    return TP / (TP + FP)


print(precision(y_test, nb))
print(precision_score(y_test, nb))
print("If the above match, you got it!")


# recall is true positives over all actual positive values
def recall(actual, preds):
    '''
    INPUT
    preds - predictions as a numpy array or pandas series
    actual - actual values as a numpy array or pandas series

    OUTPUT:
    returns the recall as a float
    '''
    TP = np.sum((preds == actual) & (preds > 0))
    FN = np.sum((preds == 0) & (actual == 1))
    return TP / (TP + FN)


print(recall(y_test, nb))
print(recall_score(y_test, nb))
print("If the above match, you got it!")


# f1_score is 2*(precision*recall)/(precision+recall))
def f1(actual, preds):
    '''
    INPUT
    preds - predictions as a numpy array or pandas series
    actual - actual values as a numpy array or pandas series

    OUTPUT:
    returns the f1score as a float
    '''
    prec = precision(actual, preds)
    rec = recall(actual, preds)
    return 2 * ((prec * rec) / (prec + rec))


print(f1(y_test, nb))
print(f1_score(y_test, nb))
print("If the above match, you got it!")


# add the letter of the most appropriate metric to each statement
# in the dictionary
a = "recall"
b = "precision"
c = "accuracy"
d = 'f1-score'


seven_sol = {
    'We have imbalanced classes, which metric do we definitely not want to'
    ' use?': c,
    'We really want to make sure the positive cases are all caught even if'
    ' that means we identify some negatives as positives': a,
    'When we identify something as positive, we want to be sure it is truly'
    ' positive': b,
    'We care equally about identifying positive and negative cases': d
}

# This gives: That's right!  It isn't really necessary to memorize these in
# practice, but it is important to know they exist and know why might use one
# metric over another for a particular situation.


models = {'nb': nb,
          'bag_pred': bag_pred,
          'rf_pred': rf_pred,
          'ada_pred': ada_pred,
          'svm_pred': svm_pred}
metrics = [accuracy_score, precision_score, recall_score, f1_score]

for i in models:
    for j in range(len(metrics)):
        print(f'{metrics[j].__name__} for '
              f'{i} {metrics[j](y_test, models[i]):.4f}')
    print()


beta = 1

print(f1_score(y_test, nb))
print(fbeta_score(y_test, nb, beta))

for i in models:
    print(f'fbeta_score for {i} {fbeta_score(y_test, models[i], beta)}')
    print(f'f1_score for {i} {f1_score(y_test, models[i], beta)}')
    print()


# Function for calculating auc and roc

def build_roc_auc(model, X_train, X_test, y_train, y_test):
    '''
    INPUT:
    model - an sklearn instantiated model
    X_train - the training data
    y_train - the training response values (must be categorical)
    X_test - the test data
    y_test - the test response values (must be categorical)
    OUTPUT:
    auc - returns auc as a float
    prints the roc curve
    '''
    y_preds = model.fit(X_train, y_train).predict_proba(X_test)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(y_test)):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_preds[:, 1])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(),
                                              y_preds[:, 1].ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.show()

    return roc_auc_score(y_test, np.round(y_preds[:, 1]))


instaniated_models = [naive_bayes, bag_mod, rf_mod]

for i in instaniated_models:
    build_roc_auc(i, training_data, testing_data, y_train, y_test)

print(build_roc_auc(instaniated_models[0], training_data, testing_data,
                    y_train, y_test))
