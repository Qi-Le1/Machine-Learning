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
        self.weight_distribution = None
        self.alpha = None

    def train_weak_learner(self, data: Any, label: Any, weak_learner_number: int) -> None:

        weight_distribution = np.zeros(data.shape[0])
        for i in range(len(weight_distribution)):
            weight_distribution[i] = 1 / len(weight_distribution)

        self.traindata = data
        self.trainlabel = label

        self.weight_distribution = weight_distribution  # Record weight distribution
        self.max_classifier = np.zeros((weak_learner_number, 3))  # Record each decision stump
        self.alpha = np.zeros(weak_learner_number)  # Record alpha

        for i in range(weak_learner_number):
            selected_train_array = np.random.choice(data.shape[0], int(data.shape[0] * 0.6),
                                                    p=self.weight_distribution)

            self.data = data[selected_train_array]
            self.label = label[selected_train_array]

            tree = Decision_Tree()
            self.max_classifier[i] = tree.find_model(self.data, self.label)
            y_predicted = tree.predict(self.max_classifier[i], self.traindata)

            # update i-th weak learner
            self.alpha[i] = self.update_parameters(y_predicted)

    def update_parameters(self, y_predicted) -> None:

        Error = 0
        for i in range(self.trainlabel.shape[0]):
            if self.trainlabel[i] != y_predicted[i]:
                Error += self.weight_distribution[i]

        Error = max(Error, 1e-7)
        alpha = 0.5 * np.log((1 - Error) / Error)

        # Update weight distribution
        for i in range(self.trainlabel.shape[0]):
            if self.trainlabel[i] == y_predicted[i]:
                self.weight_distribution[i] = self.weight_distribution[i] / (2 * (1 - Error))
            elif self.trainlabel[i] != y_predicted[i]:
                self.weight_distribution[i] = self.weight_distribution[i] / (2 * Error)

        return alpha

    def test(self, test_data: Any, test_label: Any, ):
        y_predicted = np.zeros(test_data.shape[0])

        for i in range(self.max_classifier.shape[0]):
            tree = Decision_Tree()
            y_predicted += self.alpha[i] * tree.predict(self.max_classifier[i], test_data)

        y_predicted = np.sign(y_predicted)

        Error = 0
        for i in range(test_label.shape[0]):
            if y_predicted[i] != test_label[i]:
                Error += 1

        return (Error / test_label.shape[0])


def adaboost(dataset: str) -> None:
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

    weak_learner_number = 101
    # weak_learner_number = [5]
    split = 0.2

    X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=split, random_state=None)
    train_error_list = []
    test_error_list = []
    for i in range(1, weak_learner_number):
        adaboost = Adaboost()
        adaboost.train_weak_learner(X_train, Y_train, i)

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

if __name__ == "__main__":
    dataset = ("C:\\Users\\Lucky\\PycharmProjects\\ml_hw4\\breast-cancer-wisconsin.data")
    adaboost(dataset)
