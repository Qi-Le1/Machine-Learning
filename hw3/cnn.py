import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
import numpy as np
import matplotlib.pyplot as plt
import time

class CNN:

    def train(self, x_train, y_train, batch, first, optimizer_number):
        # Define the model.
        self.model = Sequential()

        self.model.add(Conv2D(20, kernel_size=(3, 3), activation='relu', strides=(1, 1), input_shape=(28, 28, 1)))
        self.model.add(MaxPool2D((2, 2)))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu', use_bias=True))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation='softmax', use_bias=False))

        optimizer_list = ["SGD", "Adagrad", "Adam"]

        # # Compile the model.
        self.model.compile(optimizer=optimizer_list[optimizer_number], loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # # Fit the model.

        epochs_number = 15
        convergence = []
        for i in range(len(batch)):
            time_start = time.time()
            train_result = self.model.fit(x_train, y_train, epochs=epochs_number, batch_size=batch[i], verbose=1,
                                          validation_split=0.1, shuffle=True)
            time_end = time.time()
            convergence_run_time = (time_end - time_start) / epochs_number
            convergence.append(convergence_run_time)

        if first == 1:
            plt.plot(train_result.history['loss'], color='green', marker='o', label="Training Set Loss")
            plt.plot(train_result.history['accuracy'], color='blue', marker='o', label="Training Set Accuracy")
            plt.plot(train_result.history['val_loss'], color='red', marker='o', label="Validation Set Loss")
            plt.plot(train_result.history['val_accuracy'], color='yellow', marker='o', label="Validation Set Accuracy")
            plt.xlabel('Epoch Numbers', fontsize=12)
            plt.ylabel('Percentage', fontsize=12)
            plt.legend(loc='right', fontsize=12)
            plt.show()
        else:
            plt.plot(batch,convergence, color='green', marker='o', label=optimizer_list[optimizer_number])
            plt.xlabel('Mini Batch Size', fontsize=12)
            plt.ylabel('Convergence Run Time(s)', fontsize=12)
            plt.legend(loc='right', fontsize=12)
            plt.show()

    def test(self, x_test, y_test):
        predict_result = self.model.predict(x_test).reshape((x_test.shape[0], 10)).argmax(axis=1)
        correct = 0
        for i in tf.equal(predict_result, y_test):
            if i == True:
                correct += 1
        return correct / x_test.shape[0]


def cnn() -> None:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

    # Normalize Data
    x_train = x_train / 255.0
    x_train_mean = np.mean(x_train, axis=0)
    x_train_variance = np.std(x_train, axis=0) + 1e-9
    x_train = (x_train - x_train_mean) / x_train_variance

    x_test = x_test / 255.0
    x_test_mean = np.mean(x_test, axis=0)
    x_test_variance = np.std(x_test, axis=0) + 1e-9
    x_test = (x_test - x_test_mean) / x_test_variance

    # Evaluate the model for the first part
    batch = [32]
    multi_layer = CNN()
    multi_layer.train(x_train, y_train, batch, 1, 0)
    accuracy = multi_layer.test(x_test, y_test)
    print("Testing set accuracy:", accuracy)

    # Evaluate the model for the second part
    batch = [32, 64, 96, 128]
    for i in range(3):
        multi_layer.train(x_train, y_train, batch, 2, i)

    return None


if __name__ == "__main__":
    cnn()
