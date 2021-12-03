import sklearn as sk    #import dataset
import numpy as np
import statistics
from sklearn import datasets  #import dataset
import sys

X, t = sk.datasets.load_boston(return_X_y=True)

t_copy = list(t)
t_copy.sort()

label = []
shape = (13,1)  # The shape of each X

median = np.median(t_copy)    # Fine the median number
seventy_five_percentile = np.percentile(t_copy, 75) # Find the 75th percentile number

for i in range(506):
    if t[i] >= median:
        label.append(1)  #When the t number >= median, add the corresponding X to the list
    else:
        label.append(0)  #When the t number < median, add the corresponding X to the list


def cut(X, label, testing_set_start, testing_set_end):  # Devide the training set and testing set
    label_1 = []
    label_0 = []
    testing_set = []

    for i in range(0, testing_set_start):  # training set
        if label[i] == 1:
            label_1.append(X[i].reshape(shape))  # When the label number == 1, add the corresponding X to the list
        else:
            label_0.append(X[i].reshape(shape))  # When the label number != 1, add the corresponding X to the list

    for i in range(testing_set_end, len(X)):  # training set
        if (i >= len(X)):
            break
        if label[i] == 1:
            label_1.append(X[i].reshape(shape))  # When the label number == 1, add the corresponding X to the list
        else:
            label_0.append(X[i].reshape(shape))  # When the label number != 1, add the corresponding X to the list

    for i in range(testing_set_start, testing_set_end):  # testing set
        if i >= len(X):
            break
        testing_set.append(X[i].reshape(shape))
    return label_1, label_0, testing_set


def mean(label_1, label_0):
    label_1_mean = 0  # Assume the X with label 1 is c1 part
    label_0_mean = 0  # Assume the X with label 0 is c2 part

    label_1_sum = 0
    label_0_sum = 0

    for i in label_1:
        label_1_sum += i
    label_1_mean = label_1_sum / len(label_1)

    for i in label_0:
        label_0_sum += i
    label_0_mean = label_0_sum / len(label_0)
    return label_1_mean, label_0_mean, label_1_sum, label_0_sum


def Sb_cal(label_1_mean, label_0_mean):
    # Sb = (Boston50_label_1_mean - Boston50_label_0_mean)(Boston50_label_1_mean - Boston50_label_0_mean)'
    difference_mean = label_1_mean - label_0_mean
    difference_mean_transpose = np.transpose(difference_mean)
    Sb = np.dot(difference_mean, difference_mean_transpose)
    return Sb


def Sw_cal(label_1, label_0, label_1_mean, label_0_mean):
    # Sw = Sc1+Sc2 = covariance of c1 + covariance of c2
    Sc1 = 0
    Sc2 = 0

    difference = 0
    difference_transpose = 0

    for i in label_1:
        difference = i - label_1_mean
        difference_transpose = np.transpose(difference)
        Sc1 += np.dot(difference, difference_transpose)

    for i in label_0:
        difference = i - label_0_mean
        difference_transpose = np.transpose(difference)
        Sc2 += np.dot(difference, difference_transpose)

    Sw = Sc1 + Sc2
    return Sw


def direction_w(Sw, label_1_mean, label_0_mean):
    # The direction of w is propotional to (Sw)^(-1) * (c1_mean - c2_mean)   Formula 4.38
    Sw_inverse = np.linalg.inv(Sw)
    w = np.dot(Sw_inverse, (label_1_mean - label_0_mean))
    return w


def w0_cal(w, label_1_sum, label_0_sum, X):
    # wo = -(w)'m Formula 4.34

    w_transpose = np.transpose(w)
    m = (label_1_sum + label_0_sum) / len(X)  # m = Sum of the two parts/the number total data points
    w0 = -(np.dot(w_transpose, m))
    return w0


def main(cross_number):

    k = 0
    j = 0
    w = 0

    every_set = round(len(X) / int(cross_number))
    test_error_list = []
    train_error_list = []

    for i in range(1, int(cross_number) + 1):

        train_correct = 0
        test_correct = 0

        train_correct_percentage = 0
        test_correct_percentage = 0

        testing_set_start = 0 + k
        testing_set_end = every_set + k

        testing_data_number = 0

        Boston50_label_1, Boston50_label_0, testing_set = cut(X, label, testing_set_start, testing_set_end)

        Boston50_label_1_mean, Boston50_label_0_mean, Boston50_label_1_sum, Boston50_label_0_sum = mean(
            Boston50_label_1, Boston50_label_0)

        Sb = Sb_cal(Boston50_label_1_mean, Boston50_label_0_mean)

        Sw = Sw_cal(Boston50_label_1, Boston50_label_0, Boston50_label_1_mean, Boston50_label_0_mean)

        w = direction_w(Sw, Boston50_label_1_mean, Boston50_label_0_mean)
        w_transpose = np.transpose(w)

        w0 = w0_cal(w, Boston50_label_1_sum, Boston50_label_0_sum, X)

        for j in range(testing_set_start,testing_set_end):  # Devide the group into 51,51,51,51,51,51,51,51,51,47, test the testing set
            if j >= len(X):
                break

            testing_result = np.dot(w_transpose, X[j]) + w0

            if (testing_result >= 0):
                if (label[j] == 1):
                    test_correct += 1
            else:
                if (label[j] == 0):
                    test_correct += 1

            testing_data_number += 1

        for training in Boston50_label_1:
            training_result = np.dot(w_transpose, training) + w0
            if (training_result >= 0):
                train_correct += 1

        for training in Boston50_label_0:
            training_result = np.dot(w_transpose, training) + w0
            if (training_result < 0):
                train_correct += 1

        test_correct_percentage = test_correct / testing_data_number
        train_correct_percentage = train_correct / (len(X) - testing_data_number)
        test_error_percentage = 1 - test_correct_percentage
        train_error_percentage = 1 - train_correct_percentage

        train_error_list.append(train_error_percentage)
        test_error_list.append(test_error_percentage)

        k = i * every_set

    print('train_error_list: ',train_error_list)
    print('standard deviation of train_error_list: ',statistics.stdev(train_error_list))
    print('test_error_list: ',test_error_list)
    print('standard deviation of test_error_list: ',statistics.stdev(test_error_list))



if __name__ == "__main__":
    main(sys.argv[1])


