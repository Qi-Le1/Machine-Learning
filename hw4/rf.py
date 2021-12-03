import numpy as np
import pandas
import math
from typing import Any
from sklearn.model_selection import train_test_split
from decision_tree import Decision_Tree
import matplotlib.pyplot as plt


class Adaboost():
    def __init__(self):
        self.traindata = None
        self.trainlabel = None
        self.data = None
        self.label = None
        self.max_classifier = None
        self.feature = None

    def train_weak_learner(self, data: Any, label: Any, weak_learner_number: int, feature_count: int) -> None:
        self.traindata = data
        self.trainlabel = label

        self.max_classifier = np.zeros((weak_learner_number, 3))  # Record each decision stump
        self.feature = np.zeros((weak_learner_number, feature_count))

        for i in range(weak_learner_number):
            selected_train_array = np.random.choice(data.shape[0], int(data.shape[0] * 0.6))
            selected_feature = np.random.choice(data.shape[1], feature_count)
            self.data = data[selected_train_array][:, selected_feature]
            self.label = label[selected_train_array]

            tree = Decision_Tree()
            self.max_classifier[i] = tree.find_model(self.data, self.label)
            self.feature[i] = selected_feature

        return

    def test(self, test_data: Any, test_label: Any):
        y_predicted = np.zeros(test_data.shape[0])

        for i in range(self.max_classifier.shape[0]):
            tree = Decision_Tree()
            y_predicted += tree.predict(self.max_classifier[i], test_data[:, self.feature[i].astype(int)])

        y_predicted = np.sign(y_predicted)

        Error = 0
        for i in range(test_label.shape[0]):
            if y_predicted[i] != test_label[i]:
                Error += 1

        return (Error / test_label.shape[0])


def fixed_attributes(X_train, X_test, Y_train, Y_test):
    weak_learner_number = 101
    train_error_list = []
    test_error_list = []
    feature_count = 3

    for i in range(1, weak_learner_number):
        adaboost = Adaboost()

        adaboost.train_weak_learner(X_train, Y_train, i, feature_count)
        train_error_rate = adaboost.test(X_train, Y_train)
        test_error_rate = adaboost.test(X_test, Y_test)

        train_error_list.append(train_error_rate)
        test_error_list.append(test_error_rate)

    plt.plot(train_error_list, color='green', marker='o', label="train_error")
    plt.plot(test_error_list, color='red', marker='o', label="test_error")
    plt.xlabel('Number of Weak Learners', fontsize=12)
    plt.ylabel('Error Rate', fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.show()


def fixed_decision_stumps(X_train, X_test, Y_train, Y_test):
    train_error_list = []
    test_error_list = []
    weak_learner_number = 100

    for i in range(2, 10):
        adaboost = Adaboost()

        adaboost.train_weak_learner(X_train, Y_train, weak_learner_number, i)
        train_error_rate = adaboost.test(X_train, Y_train)
        test_error_rate = adaboost.test(X_test, Y_test)

        train_error_list.append(train_error_rate)
        test_error_list.append(test_error_rate)

    plt.plot(train_error_list, color='green', marker='o', label="train_error")
    plt.plot(test_error_list, color='red', marker='o', label="test_error")
    plt.xlabel('Number of Attributes', fontsize=12)
    plt.ylabel('Error Rate', fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.show()

def rf(dataset: str) -> None:
    document = pandas.read_csv(dataset, header=None)
    data = document.iloc[:, 1:10].to_numpy()  # data points array
    label = document.iloc[:, 10].to_numpy()

    drop_row = []

    for i in range(data.shape[0]):

        if label[i] == 2:
            label[i] = -1
        else:
            label[i] = 1

        if data[i][5] == '?':
            drop_row.append(i)
            continue
        else:
            data[i][5] = int(data[i][5])

    data = np.delete(data, drop_row, axis=0)
    label = np.delete(label, drop_row, axis=0)

    split = 0.2
    X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=split, random_state=None)

    #Comment the one you do not want to compute
    #fixed_attributes(X_train, X_test, Y_train, Y_test)
    fixed_decision_stumps(X_train, X_test, Y_train, Y_test)


if __name__ == "__main__":
    dataset = ("C:\\Users\\Lucky\\PycharmProjects\\ml_hw4\\breast-cancer-wisconsin.data")
    rf(dataset)
