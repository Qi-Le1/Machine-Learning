import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
import numpy as np
import matplotlib.pyplot as plt

class multilayer:

    def train(self, x_train, y_train):
        # Define the model.
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(28, 28, 1)))
        self.model.add(Dense(128, activation='relu', use_bias=True))
        self.model.add(Dense(10, activation='softmax', use_bias=False))

        # # Compile the model.
        self.model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # # Fit the model.
        train_result = self.model.fit(x_train, y_train, epochs=15, batch_size=32, verbose=1, validation_split=0.1, shuffle=True)
        plt.plot(train_result.history['loss'],color='green', marker='o',label = "Training Set Loss")
        plt.plot(train_result.history['accuracy'],color='blue', marker='o',label = "Training Set Accuracy")
        plt.plot(train_result.history['val_loss'],color='red', marker='o',label = "Validation Set Loss")
        plt.plot(train_result.history['val_accuracy'],color='yellow', marker='o',label = "Validation Set Accuracy")
        plt.xlabel('Epoch Numbers', fontsize=12)
        plt.ylabel('Percentage', fontsize=12)
        plt.legend(loc='right', fontsize=12)
        plt.show()

    def test(self, x_test, y_test):
        predict_result = self.model.predict(x_test).reshape((x_test.shape[0], 10)).argmax(axis=1)
        correct = 0
        for i in tf.equal(predict_result, y_test):
            if i == True:
                correct += 1
        return correct / x_test.shape[0]


def neural_net() -> None:

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

    multi_layer = multilayer()
    multi_layer.train(x_train, y_train)

    # Evaluate the model
    accuracy = multi_layer.test(x_test, y_test)
    print("Testing set accuracy:", accuracy)

    return None

if __name__ == "__main__":
    neural_net()
