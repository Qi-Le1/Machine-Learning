import numpy as np
import pandas

from cvxopt import matrix, solvers
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

class Support_Vector_Machine():

    def __init__(self):
        self.w_optimal = None
        self.b = None

    def fit(self,X,Y,C):
        # setting the parameters for the QP optimization
        Y = Y.reshape(-1,1)
        K = np.dot(X,X.T)
        P = matrix(np.outer(Y,Y)*K)
        q = matrix(-np.ones((X.shape[0],1)))
        G = matrix(np.concatenate((-np.identity(X.shape[0]), np.identity(X.shape[0])), axis=0), tc='d')
        h = matrix(np.concatenate((np.zeros((X.shape[0], 1)), C * np.ones((X.shape[0], 1))), axis=0), tc='d')
        A = matrix(Y.reshape(1,-1))
        b = matrix(np.zeros(1))

        max_lambda = solvers.qp(P,q,G,h,A,b)
        lambda_value = max_lambda['x']

        lambda_value = np.squeeze(np.asarray(lambda_value))
        larger_zero_indices = lambda_value > 1e-30
        lambda_for_w = lambda_value[larger_zero_indices]
        # compute support vector
        X_support_vector = X[larger_zero_indices]
        # Get the corresponding labels
        Y_support_vector = Y[larger_zero_indices]
        larger_zero_indices = np.arange(len(X))[larger_zero_indices]

        self.w_optimal = np.zeros((1,len(lambda_for_w)))    #calculate w
        self.w_optimal = np.dot((lambda_for_w.reshape((len(lambda_for_w),1))*Y_support_vector).T,X_support_vector)


        #calculate b
        self.b = 0.0
        count = 0
        for i in range(len(lambda_for_w)):
            self.b += Y_support_vector[i] - np.sum(lambda_for_w[i]*Y_support_vector[i]*K[larger_zero_indices[i],larger_zero_indices])
            count += 1
        self.b /= count

        return self

    def predict(self,X):
        prediction = np.zeros((X.shape[0],1))
        for i in range(X.shape[0]):
            prediction[i] = np.sign(np.dot(self.w_optimal, X[i]) + self.b)
        return prediction

    def error_rate(self, Y_pred, Y_label):
        # prediction error_rate
        sum = 0
        for i in range(len(Y_pred)):
            if (Y_pred[i] != Y_label[i]):
                sum += 1
        return sum / len(Y_pred)

def circulation(X,Y,k_fold_cross_validation,split,C):

    # 80/20 split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split, random_state=None)

    Result_validation = [[[0] for u1 in range(len(C))] for u2 in range(k_fold_cross_validation)]
    Result_train = [[[0] for u1 in range(len(C))] for u2 in range(k_fold_cross_validation)]

    for i in range(len(C)):
        print("This is C", C[i])
        row = 0

        kf = KFold(n_splits=k_fold_cross_validation, shuffle=True, random_state=None)
        for X_train_train_index, X_train_test_index in kf.split(X_train, Y_train):
            X_train_train, Y_train_train = X_train[X_train_train_index], Y_train[X_train_train_index]
            X_train_test, Y_train_test = X_train[X_train_test_index], Y_train[X_train_test_index]

            SVM = Support_Vector_Machine()

            SVM.fit(X_train_train,Y_train_train,C[i])

            predict_train_set = SVM.predict(X_train_train)
            predict_validation = SVM.predict(X_train_test)

            Errorrate_train = SVM.error_rate(predict_train_set, Y_train_train)
            Errorrate_validation = SVM.error_rate(predict_validation,Y_train_test)

            Result_train[row][i] = Errorrate_train
            Result_validation[row][i] = Errorrate_validation
            row += 1

    average_train_error_rate = np.mean(Result_train, axis=0)
    Std_train_error_rate = np.std(Result_train,axis = 0)
    average_validation_error_rate = np.mean(Result_validation, axis=0)
    Std_validation_error_rate = np.std(Result_validation, axis=0)

    #list_validation = average_validation_error_rate.tolist()
    min_validation_error = 100
    min_index = 0
    for i in range(len(average_validation_error_rate)-1,-1,-1):   #From back to front
        if average_validation_error_rate[i] < min_validation_error:
            min_validation_error = average_validation_error_rate[i]
            min_index = i
    chosen_C = C[min_index]

    # Assume we have C[C1,C2,C3], the store would be like C1,C2,C3.
    print("Average_train_error_rate", average_train_error_rate)
    print("Std_train_error_rate", Std_train_error_rate)
    print("Average_validation_error_rate", average_validation_error_rate)
    print("Std_validation_error_rate", Std_validation_error_rate)
    print("chosen_C",chosen_C)

    # Fit the best parameter
    SVM.fit(X_train,Y_train,chosen_C)
    best_test_predict = SVM.predict(X_test)
    best_test_errorrate = SVM.error_rate(best_test_predict,Y_test)

    print("Error rate for the best model: ", best_test_errorrate)


def SVM_dual(dataset: str) -> None:
    document = pandas.read_csv(dataset, header=None)

    # Regularization
    data = document.iloc[:, 0:7].to_numpy()  # data points array
    data_mean = np.mean(data, axis=0)
    data_variance = np.std(data, axis=0) + 1e-9
    data = (data - data_mean) / data_variance

    label = document.iloc[:, 7].to_numpy()  # labels array

    for i in range(len(label)):
        if label[i] == 0:
            label[i] = -1
    C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    circulation(data, label, 10, 0.2, C)

    return

if __name__ == "__main__":
    SVM_dual("C:\\Users\\Lucky\\PycharmProjects\\hw2\\hw2_data_2020.csv")
