import numpy as np
import math

class Decision_Tree():
    def __init__(self):
        self.row = 0
        self.column = 0
        self.data = None
        self.label = None

    def predict(self, parameter, data):
        row = parameter[0]
        column = int(parameter[1])
        left_side = parameter[2]

        y = (data[:, column] <= row)
        y = (left_side * np.array([-1, 1])).take(y, axis=0)

        return y

    def left_right(self):
        left_side = -1
        predicted_y_neg = (self.data[:, self.column] <= self.row)
        predicted_y_neg = (left_side * np.array([-1, 1])).take(predicted_y_neg, axis=0)
        negative = np.mean(predicted_y_neg == self.label)

        left_side = 1
        predicted_y_pos = (self.data[:, self.column] <= self.row)
        predicted_y_pos = (left_side * np.array([-1, 1])).take(predicted_y_pos, axis=0)
        positive = np.mean(predicted_y_pos == self.label)

        if negative >= positive:
            left_side = -1
        else:
            left_side = 1

        return left_side

    def find_model(self, data, label):
        self.data = data
        self.label = label

        information_gain = -100

        for i in range(11):
            for j in range(self.data.shape[1]):
                information_gain_ij = self.information_gain(i, j)
                if information_gain_ij > information_gain:
                    information_gain = information_gain_ij
                    self.row = i
                    self.column = j

        left_side = self.left_right()
        return self.row, self.column, left_side

    def information_gain(self, number: int, column: int):
        negative_count = 0
        positive_count = 0
        negative_class = []
        positive_class = []

        for i in range(self.data.shape[0]):
            if self.label[i] == -1:
                negative_count += 1
            elif self.label[i] == 1:
                positive_count += 1

            if self.data[i][column] <= number:
                negative_class.append(self.label[i])
            else:
                positive_class.append(self.label[i])

        negative_class_negative = 0
        negative_class_positive = 0
        for i in negative_class:
            if i == -1:
                negative_class_negative += 1
            else:
                negative_class_positive += 1

        positive_class_negative = 0
        positive_class_positive = 0
        for i in positive_class:
            if i == -1:
                positive_class_negative += 1
            else:
                positive_class_positive += 1

        # The first term of the information gain equation
        Entropy_total = 0
        if negative_count != 0:
            Entropy_total -= (negative_count / self.label.shape[0]) * math.log2(negative_count / self.label.shape[0])

        if positive_count != 0:
            Entropy_total -= (positive_count / self.label.shape[0]) * math.log2(positive_count / self.label.shape[0])

        # The Second term of the information gain equation
        Entropy_i = 0
        coefficient1 = len(negative_class) / len(self.label)
        coefficient2 = len(positive_class) / len(self.label)

        if len(negative_class) != 0:
            d11 = negative_class_negative / len(negative_class)
            d12 = negative_class_positive / len(negative_class)

            if negative_class_negative != 0:
                d11 = d11 * math.log2(d11)
            if negative_class_positive != 0:
                d12 = d12 * math.log2(d12)
            Entropy_i += coefficient1 * (-(d11 + d12))

        if len(positive_class) != 0:
            d21 = positive_class_negative / len(positive_class)
            d22 = positive_class_positive / len(positive_class)

            if positive_class_negative != 0:
                d21 = d21 * math.log2(d21)
            if positive_class_positive != 0:
                d22 = d22 * math.log2(d22)

            Entropy_i += coefficient2 * (-(d21 + d22))

        return (Entropy_total - Entropy_i)
