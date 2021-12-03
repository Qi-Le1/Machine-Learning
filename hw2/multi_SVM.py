import numpy as np
import pandas
from cvxopt import matrix, solvers
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

class Support_Vector_Machine():

    def RBF_kernel(self,X1,X2):

        if X1.shape[0] == 1 and X2.shape[0] == 1:                            #If X1 and X2 is a single vector
            nominator = (np.linalg.norm(X1 - X2) ** 2)
            RBF = np.exp(-nominator / (2 * self.sigma * self.sigma))
            return RBF
        elif X1.shape[0] == X2.shape[0]:                                     #Matrix = Matrix.T
            RBF = np.zeros((X1.shape[0],X2.shape[0]))
            for i in range(X1.shape[0]):
                for j in range(i+1):
                    nominator = np.linalg.norm(X1[i] - X2[j])**2
                    K = np.exp(-nominator/(2*self.sigma*self.sigma))
                    RBF[i][j] = K
                    RBF[j][i] = K
            return RBF
        else:                                                                #Rows are different
            RBF = np.zeros((X1.shape[0],X2.shape[0]))
            for i in range(X1.shape[0]):
                for j in range(X2.shape[0]):
                    nominator = np.linalg.norm(X1[i] - X2[j])**2
                    K = np.exp(-nominator/(2*self.sigma*self.sigma))
                    RBF[i][j] = K
            return RBF

    def fit(self,X,Y,C,sigma1,kernel):
        self.sigma = sigma1

        Y_copy = Y.copy()
        #print("nini",Y_copy)
        w_list = []
        b_list = []
        X_support = []
        Y_support = []

        if kernel == "RBF Kernel":
            K = self.RBF_kernel(X, X)
        elif kernel == "Linear Kernel":
            K = np.dot(X, X.T)

        for i in range(10):
            for j in range(len(Y)):
                if Y[j] == i:
                    Y_copy[j] = 1
                else:
                    Y_copy[j] = -1

            Y_copy = Y_copy.reshape(-1,1)     #setting the parameters for the QP optimization

            P = matrix(np.outer(Y_copy, Y_copy) * K)
            q = matrix(-np.ones((X.shape[0],1)))
            G = matrix(np.concatenate((-np.identity(X.shape[0]), np.identity(X.shape[0])), axis=0), tc='d')
            h = matrix(np.concatenate((np.zeros((X.shape[0], 1)), C * np.ones((X.shape[0], 1))), axis=0), tc='d')
            A = matrix(Y_copy.reshape(1,-1))
            b = matrix(np.zeros(1))

            max_lambda = solvers.qp(P,q,G,h,A,b)
            lambda_value = max_lambda['x']
            lambda_value = np.squeeze(np.asarray(lambda_value))
            larger_zero_indices = lambda_value > 1e-40

            self.lambda_for_w = lambda_value[larger_zero_indices]

            # compute support vector
            self.X_support_vector = X[larger_zero_indices]
            # Get the corresponding labels
            self.Y_support_vector = Y_copy[larger_zero_indices]
            larger_zero_indices = np.arange(len(X))[larger_zero_indices]

            self.w_optimal = np.zeros((1, len(self.lambda_for_w)))  # calculate w for linear kernel
            self.b = 0.0

            if kernel == "RBF Kernel":                             #b for RBF kernel

                count = 0
                for i in range(len(self.lambda_for_w)):
                    self.b += self.Y_support_vector[i] - np.sum(
                        self.lambda_for_w[i] * self.Y_support_vector[i] * K[
                            larger_zero_indices[i], larger_zero_indices])
                    count += 1
                self.b /= count

                w_list.append(self.lambda_for_w)
                b_list.append(self.b)
                X_support.append(self.X_support_vector)
                Y_support.append(self.Y_support_vector)

            elif kernel == "Linear Kernel":                        #w and b for linear kernel
                self.w_optimal = np.dot((self.lambda_for_w.reshape((len(self.lambda_for_w), 1)) * self.Y_support_vector).T,
                                        self.X_support_vector)

                count = 0
                for i in range(len(self.lambda_for_w)):
                    self.b += self.Y_support_vector[i] - np.sum(
                        self.lambda_for_w[i] * self.Y_support_vector[i] * K[
                            larger_zero_indices[i], larger_zero_indices])
                    count += 1
                self.b /= count

                w_list.append(self.w_optimal)
                b_list.append(self.b)

        return w_list,b_list,X_support,Y_support

    def predict(self,X,kernel,w,b,X_support,Y_support):

        if kernel == "RBF Kernel":
            prediction = np.zeros((10,X.shape[0]))
            for i in range(10):
                onenumebr_prediction = np.sum(Y_support[i] * w[i].reshape((len(w[i]),1)) * self.RBF_kernel(X_support[i], X),
                                    axis=0)
                bias = np.ones_like(onenumebr_prediction) * b[i]
                prediction[i] = onenumebr_prediction + bias
            prediction = np.argmax(prediction,axis=0)

        elif kernel == "Linear Kernel":
            prediction = np.zeros((X.shape[0], 1))
            for i in range(X.shape[0]):
                single_prediction = [[] for zz in range(10)]
                for j in range(10):
                    single_prediction[j] = np.dot(w[j], X[i]) + b[j]
                prediction[i] = single_prediction.index(max(single_prediction))

        return prediction

    def error_rate(self, Y_pred, Y_label):
        # prediction error_rate
        sum = 0
        for i in range(len(Y_pred)):
            if (Y_pred[i] != Y_label[i]):
                sum += 1
        return sum / len(Y_pred)

def circulation(X,Y,k_fold_cross_validation,split,C,sigma,kernel):

    # 80/20 split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split, random_state=None)
    if kernel == "RBF Kernel":
        Result_validation = [[[0] for u1 in range(len(C) * len(sigma))] for u2 in range(k_fold_cross_validation)]
        Result_train = [[[0] for u1 in range(len(C) * len(sigma))] for u2 in range(k_fold_cross_validation)]
    elif kernel == "Linear Kernel":
        Result_validation = [[[0] for u1 in range(len(C))] for u2 in range(k_fold_cross_validation)]
        Result_train = [[[0] for u1 in range(len(C))] for u2 in range(k_fold_cross_validation)]

    for i in range(len(C)):
        column = i
        if kernel == "RBF Kernel":
            for j in range(len(sigma)):
                row = 0
                kf = KFold(n_splits=k_fold_cross_validation, shuffle=True, random_state=None)
                for X_train_train_index, X_train_test_index in kf.split(X_train, Y_train):
                    X_train_train, Y_train_train = X_train[X_train_train_index], Y_train[X_train_train_index]
                    X_train_test, Y_train_test = X_train[X_train_test_index], Y_train[X_train_test_index]
                    SVM = Support_Vector_Machine()

                    w,b,X_support,Y_support = SVM.fit(X_train_train, Y_train_train, C[i], sigma[j], kernel)

                    predict_train_set = SVM.predict(X_train_train, kernel,w,b,X_support,Y_support)
                    predict_validation = SVM.predict(X_train_test, kernel,w,b,X_support,Y_support)

                    Errorrate_train = SVM.error_rate(predict_train_set, Y_train_train)
                    Errorrate_validation = SVM.error_rate(predict_validation, Y_train_test)

                    Result_train[row][column] = Errorrate_train
                    Result_validation[row][column] = Errorrate_validation
                    row += 1
                column += len(C)
        elif kernel == "Linear Kernel":
            row = 0
            kf = KFold(n_splits=k_fold_cross_validation, shuffle=True, random_state=None)
            for X_train_train_index, X_train_test_index in kf.split(X_train, Y_train):
                X_train_train, Y_train_train = X_train[X_train_train_index], Y_train[X_train_train_index]
                X_train_test, Y_train_test = X_train[X_train_test_index], Y_train[X_train_test_index]

                SVM = Support_Vector_Machine()
                w,b,X_support,Y_support = SVM.fit(X_train_train, Y_train_train, C[i], sigma[0], kernel)

                predict_train_set = SVM.predict(X_train_train, kernel,w,b,X_support,Y_support)
                predict_validation = SVM.predict(X_train_test, kernel,w,b,X_support,Y_support)

                Errorrate_train = SVM.error_rate(predict_train_set, Y_train_train)
                Errorrate_validation = SVM.error_rate(predict_validation, Y_train_test)

                Result_train[row][column] = Errorrate_train
                Result_validation[row][column] = Errorrate_validation
                row += 1

    average_train_error_rate = np.mean(Result_train, axis=0)
    Std_train_error_rate = np.std(Result_train, axis=0)
    average_validation_error_rate = np.mean(Result_validation, axis=0)
    Std_validation_error_rate = np.std(Result_validation, axis=0)

    # list_validation = average_validation_error_rate.tolist()
    min_validation_error = 100
    min_index = 0
    for i in range(len(average_validation_error_rate) - 1, -1, -1):  # From back to front
        if average_validation_error_rate[i] < min_validation_error:
            min_validation_error = average_validation_error_rate[i]
            min_index = i

    chosen_C = C[min_index % len(C)]
    chosen_sigma = sigma[min_index // len(C)]

    # For RBF Kernel
    # Assume we have C[C1,C2,C3], sigma[S1,S2,S3], the store would be like C1S1,C2S1,C3S1,C1S2,C2S2,C3S2,etc.

    # For Linear Kernel
    # Assume we have C[C1,C2,C3], the store would be like C1,C2,C3.
    print("Average_train_error_rate", average_train_error_rate)
    print("Std_train_error_rate", Std_train_error_rate)
    print("Average_validation_error_rate", average_validation_error_rate)
    print("Std_validation_error_rate", Std_validation_error_rate)

    print("chosen_C", chosen_C)
    if kernel == "RBF Kernel":
        print("chosen_sigma", chosen_sigma)

    w,b,X_support,Y_support = SVM.fit(X_train, Y_train, chosen_C, chosen_sigma, kernel)
    best_test_predict = SVM.predict(X_test, kernel,w,b,X_support,Y_support)
    best_test_errorrate = SVM.error_rate(best_test_predict, Y_test)

    print("Error rate for the best model: ", best_test_errorrate)

def multi_SVM(dataset1: str,dataset2: str,dataset3: str,dataset4: str,dataset5: str,dataset6: str) -> None:
    document1 = pandas.read_csv(dataset1, header=None)
    document2 = pandas.read_csv(dataset2, header=None)
    document3 = pandas.read_csv(dataset3, header=None)
    document4 = pandas.read_csv(dataset4, header=None)
    document5 = pandas.read_csv(dataset5, header=None)
    document6 = pandas.read_csv(dataset6, header=None)

    document1 = np.genfromtxt(document1[0])
    document2 = np.genfromtxt(document2[0])
    document3 = np.genfromtxt(document3[0])
    document4 = np.genfromtxt(document4[0])
    document5 = np.genfromtxt(document5[0])
    document6 = np.genfromtxt(document6[0])

    data = np.concatenate((document1, document2, document3, document4, document5, document6),
                          axis=1)  # data points array

    # Regularization
    data_mean = np.mean(data, axis=0)
    data_variance = np.std(data, axis=0) + 1e-9
    data = (data - data_mean) / data_variance

    label = np.zeros((2000, 1))  # labels array
    number = -1
    for i in range(0, 2000):
        if i % 200 == 0:
            number += 1
        label[i] = number

    C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    sigma = [0.001,0.1,1,10,100,1000]

    kernel = "Linear Kernel"            #Comment the one that does not need
    #kernel = "RBF Kernel"
    circulation(data, label, 10, 0.2, C, sigma, kernel)

if __name__ == "__main__":
    # I am assuming the input would be 6 data files
    multi_SVM("C:\\Users\\Lucky\\PycharmProjects\\hw2\\mfeat-fou","C:\\Users\\Lucky\\PycharmProjects\\hw2\\mfeat-fac",
              "C:\\Users\\Lucky\\PycharmProjects\\hw2\\mfeat-kar","C:\\Users\\Lucky\\PycharmProjects\\hw2\\mfeat-pix",
              "C:\\Users\\Lucky\\PycharmProjects\\hw2\\mfeat-zer","C:\\Users\\Lucky\\PycharmProjects\\hw2\\mfeat-mor")
