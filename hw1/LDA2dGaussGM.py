import sklearn as sk
import numpy as np
import statistics
import sys
from sklearn import datasets
X, t = sk.datasets.load_digits(n_class=10, return_X_y=True)
shape = (64,1)

def cut(X, label, testing_set_start, testing_set_end, test_length):  # Devide the training set and testing set

    training_data_length = len(X) - test_length

    training_data_set = np.zeros((training_data_length, 64))
    testing_data_set = np.zeros((test_length, 64))

    training_label_set = np.zeros(training_data_length)
    testing_label_set = np.zeros(test_length)

    j = 0
    for i in range(0, testing_set_start):  # training set
        if i>= training_data_length:
            break

        training_data_set[j] = X[i]
        training_label_set[j] = label[i]
        j += 1

    for i in range(testing_set_end, len(X)):  # training set
        if (i >= len(X)):
            break
        training_data_set[j] = X[i]
        training_label_set[j] = label[i]
        j += 1

    j = 0
    for i in range(testing_set_start, testing_set_end):  # testing set
        if i >= len(X):
            break
        testing_data_set[j] = X[i]
        testing_label_set[j] = label[i]
        j += 1

    return training_data_set, training_label_set, testing_data_set, testing_label_set

def Sw_and_Sb(training_data_set, train_label_set):
    X = training_data_set
    Y = train_label_set

    means = []
    for i in range(10):
        means.append(np.mean(X[Y == i], axis=0))
    Sw = np.zeros((64, 64))

    for i in range(10):
        covariance_matrix_each = np.zeros((64, 64))
        for j in range(len(X[Y == i])):
            row_num = X[Y == i][j].reshape(64, 1)
            means_shape = means[i].reshape(64, 1)
            covariance_matrix_each += np.dot((row_num - means_shape), (row_num - means_shape).T)
        Sw += covariance_matrix_each

    total_mean = np.mean(X, axis=0)

    Sb = np.zeros((64, 64))
    for i in range(10):
        difference = means[i].reshape(64, 1) - total_mean.reshape(64, 1)
        Sb += len(X[Y == i]) * np.dot(difference, difference.T)
    return Sw, Sb

def w_cal(Sw,Sb):
    if np.linalg.matrix_rank(Sw) != 64:
        epsilon = 1e-9
        eps = np.identity(64) * epsilon
        Sw = Sw + eps

    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    eigen_value_vector_pair = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]
    eigen_value_vector_pair = sorted(eigen_value_vector_pair, key=lambda k: k[0], reverse=True)
    #w = eigen_value_vector_pair[0][1].reshape(64, 1)
    w = np.hstack((eigen_value_vector_pair[0][1].reshape(64, 1), eigen_value_vector_pair[1][1].reshape(64, 1)))

    return w

def parameters(data_set,Y,w):

    pai = []
    total_length = len(data_set)
    for i in range(10):
        pai.append([len(data_set[Y == i]) / total_length])

    miu = []
    for i in range(10):
        miu.append(sum(data_set[Y == i] )/ len(data_set[Y == i]))

    dimension = (len(miu[0]))
    covariance = np.zeros((dimension, dimension))
    for i in range(10):
        each_covariance = np.zeros((dimension, dimension))
        for j in range(len(data_set[Y == i])):
            xx = data_set[Y == i][j]
            each_covariance += 1 / len(data_set) * np.dot(xx.reshape(dimension,1).astype(float)-miu[i].reshape(dimension,1).astype(float),
                                                                   (xx.reshape(dimension,1).astype(float)-miu[i].reshape(dimension,1).astype(float)).T)
        covariance += each_covariance

    return pai,miu,covariance

def error_rate(Y_pred, Y_label):
    # prediction error_rate
    loss = np.sum(np.abs(Y_pred - Y_label))/len(Y_pred)
    loss
    return loss

def prediction(pai,miu,covariance,X,Y):
    test_length = len(X)
    re_shape = len(miu[0])
    prediction = [[] for o in range(test_length)]
    for i in range(test_length):
        each_prediction = [[] for p in range(10)]
        for j in range(10):
            w_k = np.dot(np.linalg.inv(covariance), miu[j].reshape(re_shape, 1))
            intermediate = np.dot(miu[j].reshape(re_shape, 1).T, np.linalg.inv(covariance))
            wk_0 = -0.5 * np.dot(intermediate, miu[j].reshape(len(miu[0]), 1)) + np.log(pai[j])
            single_prediction = np.dot(w_k.T, X[i].reshape(re_shape, 1)) + wk_0
            each_prediction[j] = single_prediction
        prediction[i] = each_prediction.index(max(each_prediction))

    error = 0
    for i in range(len(prediction)):
        if prediction[i] != Y[i]:
            error += 1

    return error/len(X)

def main(cross_number):
    k = 0

    train_error_list = []
    test_error_list = []

    every_set = round(len(X) / int(cross_number))

    Sw = np.zeros((64, 64))
    Sb = np.zeros((64, 64))

    for time in range(1, int(cross_number) + 1):

        testing_set_start = 0 + k
        testing_set_end = every_set + k

        # separate testing set and training set
        training_data_set, train_label_set, testing_data_set, testing_label_set = cut(X, t, testing_set_start,
                                                                                      testing_set_end, every_set)

        Sw, Sb = Sw_and_Sb(training_data_set, train_label_set)

        w = w_cal(Sw,Sb)
        training_data_set = training_data_set.dot(w)
        testing_data_set = testing_data_set.dot(w)

        pai,miu,covariance = parameters(training_data_set,train_label_set,w)

        train_error_list.append(prediction(pai, miu, covariance, training_data_set, train_label_set))
        test_error_list.append(prediction(pai, miu, covariance, testing_data_set, testing_label_set))

        k = time * every_set

    print("train_error_list for LDA multiple class: " ,train_error_list)
    print("train_error standard deviation for LDA multiple class: ", statistics.stdev(train_error_list))
    print("test_error_list for LDA multiple class: ", test_error_list)
    print("test_error standard deviation for LDA multiple class: ", statistics.stdev(test_error_list))

if __name__ == "__main__":
    main(sys.argv[1])

