import sklearn as sk
import numpy as np
import matplotlib.pyplot as plot
from sklearn import datasets
import sys

X, t = sk.datasets.load_boston(return_X_y=True)
Y, u = sk.datasets.load_digits(n_class=10, return_X_y=True)

Y_mean = np.mean(Y,axis=0)
Y_variance = np.std(Y,axis=0)+1e-9
Y = (Y - Y_mean)/Y_variance

def trainTestSplit_Boston(X, t, cut, test_size):
    X_num = len(X)
    train_index = []
    for i in range(X_num):
        train_index.append(i)

    test_index = []
    test_num = int(X_num * test_size)
    size = X_num - test_num

    X_train = np.zeros((size, 13))
    t_train = np.zeros(size)

    X_test = np.zeros((test_num, 13))
    t_test = np.zeros(test_num)

    for i in range(test_num):
        randomIndex = int(np.random.uniform(0, len(train_index)))
        test_index.append(train_index[randomIndex])
        del train_index[randomIndex]

    j = 0
    for i in train_index:
        X_train[j] = X[i]
        if (t[i] >= cut):
            t_train[j] = 1
        else:
            t_train[j] = 0
        j += 1

    j = 0
    for i in test_index:
        X_test[j] = X[i]
        if (t[i] >= cut):
            t_test[j] = 1
        else:
            t_test[j] = 0
        j += 1

    return X_train, t_train, X_test, t_test


def trainTestSplit_digit(X, t, test_size):
    shape = (64, 1)

    X_num = len(X)
    train_index = []
    for i in range(X_num):
        train_index.append(i)

    test_index = []
    test_num = int(X_num * test_size)
    size = X_num - test_num

    X_train = np.zeros((size, 64))
    t_train = np.zeros(size)

    X_test = np.zeros((test_num, 64))
    t_test = np.zeros(test_num)

    for i in range(test_num):
        randomIndex = int(np.random.uniform(0, len(train_index)))
        test_index.append(train_index[randomIndex])
        del train_index[randomIndex]

    j = 0
    for i in train_index:
        X_train[j] = X[i]
        t_train[j] = t[i]
        j += 1

    j = 0
    for i in test_index:
        X_test[j] = X[i]
        t_test[j] = t[i]
        j += 1

    return X_train, t_train, X_test, t_test

def trainset_percentage(X, t, sep_ratio):
    # Separate data into training set and testing set.
    X_train_size = int(len(X) * (sep_ratio / 100))

    return X[:X_train_size], t[:X_train_size]

def sigmoid(z):  # Sigmoid Function

    for i in range(len(z)):  # prevent from overflow
        if z[i] <= -400:
            z[i] = -400

    return 1 / (1.0 + np.exp(-z))

def gradient(X, Y_label, w, b):
    # This function computes the gradient of cross entropy loss with respect to weight w and bias b.

    y_pred = sigmoid(np.dot(X, w) + b)
    pred_error = Y_label - y_pred
    w_grad = -np.dot(X.T, pred_error)
    b_grad = -np.sum(pred_error)

    return w_grad, b_grad

def multi_gradient(X, Y_label, w):

    w_grad = np.zeros((64, 10))

    row = len(X)
    chengji = np.dot(X , w)
    for i in range(len(X)):
        for j in range(10):
            if chengji[i][j] >= 200:
                chengji[i][j] = np.exp(200)
            else:
                chengji[i][j] = np.exp(chengji[i][j])

    err = chengji
    row_sum = err.sum(axis=1)
    row_repeat = np.zeros((row,10))

    for i in range(row):
        for j in range(10):
            row_repeat[i][j] = row_sum[i]

    err = err/row_repeat
    for i in range(len(X)):
        for j in range(10):
            if int(Y_label[i]) == j:
                err[i][j] = 1-err[i][j]
            else:
                err[i][j] = -err[i][j]

    w_grad = -np.dot(X.T,err)

    return w_grad


def predction_digit(Y_test_digit, w,u_test_digit ):
    predict = [[] for i in range(len(Y_test_digit))]

    for i in range(len(Y_test_digit)):
        small_predict = [[] for i in range(10)]
        for j in range(10):
            small_predict[j] = np.exp(np.dot(Y_test_digit[i,:], w[:,j]))
        predict[i] = small_predict.index(max(small_predict))

    return predict
def error_rate(Y_pred, Y_label):
    # prediction error_rate
    sum = 0
    for i in range(len(Y_pred)):
        if (Y_pred[i] != Y_label[i]):
            sum += 1
    return sum / len(Y_pred)

    return loss

def main(argv):

    evaluation_number = int(argv[1])

    vector = []
    for i in range(2,len(argv)):
        vector.append(int(argv[i]))

    median = np.median(t)  # Fine the median number
    seventy_five_percentile = np.percentile(t, 75)  # Find the 75th percentile number
    iteration = 200
    learning_rate = 0.07

    error_list_boston50 = [[[] for uu in range(len(vector))] for ui in range(evaluation_number)]
    error_list_boston75 = [[[] for uu in range(len(vector))] for ui in range(evaluation_number)]
    error_list_digit = [[[] for uu in range(len(vector))] for ui in range(evaluation_number)]

    for i in range(1, evaluation_number + 1):

        # Randomize the 80-20 split
        X_train_50, t_train_50, X_test_50, t_test_50 = trainTestSplit_Boston(X, t, median, 0.2)
        X_train_75, t_train_75, X_test_75, t_test_75 = trainTestSplit_Boston(X, t, seventy_five_percentile, 0.2)
        Y_train_digit, u_train_digit, Y_test_digit, u_test_digit = trainTestSplit_digit(Y, u, 0.2)

        vector_order = 0
        # print(X_train_50.shape)
        for j in vector:
            X_train_50_j, t_train_50_j= trainset_percentage(X_train_50, t_train_50, j)
            X_train_75_j, t_train_75_j= trainset_percentage(X_train_75, t_train_75, j)
            Y_train_digit_j, u_train_digit_j= trainset_percentage(Y_train_digit, u_train_digit, j)

            w_Boston_50 = np.zeros((13, ))
            b_Boston_50 = np.zeros((1,))

            w_Boston_75 = np.zeros((13,))
            b_Boston_75 = np.zeros((1,))

            w_digit = np.zeros((64,10))
            #b_digit = np.zeros((10,1))

            time = 1
            for o in range(iteration):

                w_grad_50, b_grad_50 = gradient(X_train_50_j, t_train_50_j, w_Boston_50, b_Boston_50)
                w_Boston_50 = w_Boston_50 - learning_rate / np.sqrt(time) * w_grad_50
                b_Boston_50 = b_Boston_50 - learning_rate / np.sqrt(time) * b_grad_50

                w_grad_75, b_grad_75 = gradient(X_train_75_j, t_train_75_j, w_Boston_75, b_Boston_75)
                w_Boston_75 = w_Boston_75 - learning_rate/np.sqrt(time) * w_grad_75
                b_Boston_75 = b_Boston_75 - learning_rate/np.sqrt(time) * b_grad_75

                w_grad_digit = multi_gradient(Y_train_digit_j, u_train_digit_j, w_digit)
                w_digit = w_digit - learning_rate / np.sqrt(time) * w_grad_digit

                time = time + 1

            pred_Boston_50 = np.round(sigmoid(np.matmul(X_test_50, w_Boston_50) + b_Boston_50)).astype(np.int)
            pred_Boston_75 = np.round(sigmoid(np.matmul(X_test_75, w_Boston_75) + b_Boston_75)).astype(np.int)
            pred_digit = predction_digit(Y_test_digit, w_digit,u_test_digit)

            error_list_boston50[i-1][vector_order] = (error_rate(pred_Boston_50, t_test_50))
            error_list_boston75[i-1][vector_order] = (error_rate(pred_Boston_75, t_test_75))
            error_list_digit[i-1][vector_order] = (error_rate(pred_digit, u_test_digit))

            vector_order += 1


    print('Error_list(each split for each training set percentage) of Logistic Regression for boston50 dataset:', error_list_boston50)
    print('Error_list(each split for each training set percentage) of Logistic Regression for boston75 dataset:', error_list_boston75)
    print('Error_list(each split for each training set percentage) of Logistic Regression for digit dataset:', error_list_digit)

    boston50_mean = np.mean(error_list_boston50, axis=0)
    boston75_mean = np.mean(error_list_boston75, axis=0)
    digit_mean = np.mean(error_list_digit, axis=0)

    print(
        "mean_error_list(mean of the test set error rates across all splits for each training set percentage) - boston50 dataset: ",
        boston50_mean)
    print(
        "mean_error_list(mean of the test set error rates across all splits for each training set percentage) - boston75 dataset: ",
        boston75_mean)
    print("mean_error_list(mean of the test set error rates across all splits for each training set percentage) - digits: ",
          digit_mean)

    print(
        "std_error_list(mean of the test set error rates across all splits for each training set percentage) - boston50 dataset: ",
        np.std(error_list_boston50, axis=0))
    print(
        "std_error_list(mean of the test set error rates across all splits for each training set percentage) - boston70 dataset: ",
        np.std(error_list_boston75, axis=0))
    print("std_error_list(mean of the test set error rates across all splits for each training set percentage) - digits: ",
          np.std(error_list_digit, axis=0))

    vector_plot = []
    for i in vector:
        vector_plot.append(i / 100)

    plot.figure()
    plot.errorbar(vector_plot, boston50_mean, yerr=np.std(error_list_boston50, axis=0), capsize=3, capthick=3)
    plot.title("Mean and Std of Boston_50 Error rates")

    plot.figure()
    plot.errorbar(vector_plot, boston75_mean, yerr=np.std(error_list_boston75, axis=0), capsize=3, capthick=3)
    plot.title("Mean and Std of Boston_75 Error rates")

    plot.figure()
    plot.errorbar(vector_plot, digit_mean, yerr=np.std(error_list_digit, axis=0), capsize=3, capthick=3)
    plot.title("Mean and Std of Digit Error rates")
    plot.show()


if __name__ == "__main__":
    main(sys.argv)
