import sklearn as sk
import numpy as np
import matplotlib.pyplot as plot
import sys
from sklearn import datasets

X, t = sk.datasets.load_boston(return_X_y=True)
Y, u = sk.datasets.load_digits(n_class=10, return_X_y=True)


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


def train_model(X_train_data, Y_train_data, target_label):
    X = X_train_data
    Y = Y_train_data
    feature_num = len(X[0])

    pai = []  #prior prob
    for i in range(target_label):
        pai.append(len(X[Y == i]) / len(X))

    miu = []   #mean
    standard_deviation = []

    for i in range(target_label):
        miu_small = []
        for j in range(feature_num):
            miu_small.append(np.mean(X[Y == i][:, j], axis=0))
        miu.append(miu_small)

    for i in range(target_label):
        standard_deviation_small = []
        for j in range(feature_num):
            standard_deviation_small.append(np.std(X[Y == i][:, j], axis=0))
        standard_deviation.append(standard_deviation_small)

    return pai, miu, standard_deviation


def predict_model(pai, miu, standard_deviation, X_data, target_label):
    data_length = len(X_data)
    prediction = [[] for j in range(data_length)]
    feature_num = len(X_data[0])

    for i in range(data_length):
        small_prediction = [[] for y in range(target_label)]
        for j in range(target_label):
            log_prior_prob = np.log(pai[j])
            log_gaussian_parameter = []
            log_gaussian_square = []

            for k in range(feature_num):
                if standard_deviation[j][k] != 0:

                    log_gaussian_parameter.append(
                        np.log(np.dot(standard_deviation[j][k], standard_deviation[j][k])) / 2)

                    parameter = 1 / np.dot(standard_deviation[j][k] ,standard_deviation[j][k])
                    log_gaussian_square.append(
                        0.5 * parameter * np.dot((X_data[i, k] - miu[j][k]), (X_data[i, k] - miu[j][k])))
                else:
                    log_gaussian_parameter.append(0)
                    log_gaussian_square.append(0)

            small_prediction[j] = log_prior_prob - sum(log_gaussian_parameter) - sum(log_gaussian_square)
        prediction[i] = small_prediction.index(max(small_prediction))
    return prediction


def error_rate(Y_pred, Y_label):
    # prediction error_rate
    sum = 0
    for i in range(len(Y_pred)):
        if(Y_pred[i]!=Y_label[i]):
            sum+=1
    return sum/len(Y_pred)

    return loss

def main(argv):

    evaluation_number = int(argv[1])

    vector = []
    for i in range(2,len(argv)):
        vector.append(int(argv[i]))

    median = np.median(t)  # Fine the median number
    seventy_five_percentile = np.percentile(t, 75)  # Find the 75th percentile number

    error_list_boston50 = [[[]for uu in range(len(vector))] for ui in range(evaluation_number)]
    error_list_boston75 = [[[]for uu in range(len(vector))] for ui in range(evaluation_number)]
    error_list_digit = [[[]for uu in range(len(vector))] for ui in range(evaluation_number)]

    for i in range(evaluation_number):
        # Randomize the 80-20 split
        X_train_50, t_train_50, X_test_50, t_test_50 = trainTestSplit_Boston(X, t, median, 0.2)
        X_train_75, t_train_75, X_test_75, t_test_75 = trainTestSplit_Boston(X, t, seventy_five_percentile, 0.2)
        Y_train_digit, u_train_digit, Y_test_digit, u_test_digit = trainTestSplit_digit(Y, u, 0.2)

        vector_order = 0

        # print(X_train_50.shape)
        for j in vector:
            X_train_50_j, t_train_50_j = trainset_percentage(X_train_50, t_train_50, j)
            X_train_75_j, t_train_75_j = trainset_percentage(X_train_75, t_train_75, j)
            Y_train_digit_j, u_train_digit_j = trainset_percentage(Y_train_digit, u_train_digit, j)

            pai_boston_50, miu_boston_50, standard_deviation_boston_50 = train_model(X_train_50_j, t_train_50_j, 2)
            pai_boston_75, miu_boston_75, standard_deviation_boston_75 = train_model(X_train_75_j, t_train_75_j, 2)
            pai__digit, miu_digit, standard_deviation_digit = train_model(Y_train_digit_j, u_train_digit_j, 10)

            pred_Boston_50 = predict_model(pai_boston_50, miu_boston_50, standard_deviation_boston_50, X_test_50, 2)
            pred_Boston_75 = predict_model(pai_boston_75, miu_boston_75, standard_deviation_boston_75, X_test_75, 2)
            pred_digit = predict_model(pai__digit, miu_digit, standard_deviation_digit, Y_test_digit, 10)

            error_list_boston50[i][vector_order] = (error_rate(pred_Boston_50, t_test_50))
            error_list_boston75[i][vector_order] = (error_rate(pred_Boston_75, t_test_75))
            error_list_digit[i][vector_order] = (error_rate(pred_digit, u_test_digit))

            vector_order+=1
    print("error_list(each split for each training set percentage) of Naive_Bayes for boston50 dataset: ", error_list_boston50)
    print("error_list(each split for each training set percentage) of Naive_Bayes for boston75 dataset: ", error_list_boston75)
    print("error_list(each split for each training set percentage) of Naive_Bayes for digits dataset: ", error_list_digit)

    boston50_mean = np.mean(error_list_boston50,axis=0)
    boston75_mean = np.mean(error_list_boston75,axis=0)
    digit_mean = np.mean(error_list_digit,axis=0)

    print("mean_error_list(mean of the test set error rates across all splits for each training set percentage) - boston50 dataset: ", boston50_mean)
    print("mean_error_list(mean of the test set error rates across all splits for each training set percentage) - boston70 dataset: ", boston75_mean)
    print("mean_error_list(mean of the test set error rates across all splits for each training set percentage) - digits: ", digit_mean)

    print("std_error_list(mean of the test set error rates across all splits for each training set percentage) - boston50 dataset: ", np.std(error_list_boston50,axis=0))
    print("std_error_list(mean of the test set error rates across all splits for each training set percentage) - boston70 dataset: ", np.std(error_list_boston75,axis=0))
    print("std_error_list(mean of the test set error rates across all splits for each training set percentage) - digits: ", np.std(error_list_digit,axis=0))

    vector_plot = []
    for i in vector:
        vector_plot.append(i/100)

    plot.figure()
    plot.errorbar(vector_plot,boston50_mean,yerr=np.std(error_list_boston50,axis=0),capsize=3, capthick=3)
    plot.title("Mean and Std of Boston_50 Error rates")

    plot.figure()
    plot.errorbar(vector_plot,boston75_mean,yerr=np.std(error_list_boston75,axis=0),capsize=3, capthick=3)
    plot.title("Mean and Std of Boston_75 Error rates")

    plot.figure()
    plot.errorbar(vector_plot,digit_mean,yerr=np.std(error_list_digit,axis=0),capsize=3, capthick=3)
    plot.title("Mean and Std of Digit Error rates")
    plot.show()

if __name__ == "__main__":
    main(sys.argv)


