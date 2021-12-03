import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import pylab


class K_means():
    def __init__(self):
        self.data = None
        self.k = None

    def Distortion_measure(self, x1, x2):
        return np.linalg.norm(x1 - x2)*np.linalg.norm(x1 - x2)

    def init_Centroid(self):
        row = self.data.shape[0]
        column = self.data.shape[1]

        centroid = np.zeros((self.k, column))
        index = np.random.choice(row, self.k, replace=False)
        for i in range(len(index)):
            centroid[i, :] = self.data[index[i], :]
        return centroid

    def calculation(self, data, k, iteration, acc):
        self.data = data
        self.k = k

        # init centroids
        centroid = self.init_Centroid()

        # Record the class that this point belongs to - 1st column
        # Record the distance                         - 2nd column
        Point_data = np.zeros((self.data.shape[0], 2))
        for i in range(self.data.shape[0]):
            Point_data[i][1] = 1000000

        # Find the nearest centroid for each point
        accuracy = 0
        Error_list = []
        for i in range(iteration):

            if accuracy >= acc:
                break

            wrong_class = 0
            for j in range(self.data.shape[0]):
                for z in range(k):
                    distance = self.Distortion_measure(self.data[j], centroid[z])
                    if distance < Point_data[j][1]:
                        Point_data[j][1] = distance
                        Point_data[j][0] = z
                        wrong_class += 1

            accuracy = 1 - (wrong_class / self.data.shape[0])

            # Update the position of centroid for each class
            sum_each_class = np.zeros((k, 3))
            number_each_class = np.zeros(k)

            for i in range(self.data.shape[0]):
                current_class = int(Point_data[i][0])
                sum_each_class[current_class] += self.data[i]
                number_each_class[current_class] += 1

            for i in range(k):
                centroid[i] = sum_each_class[i] / number_each_class[i]

            Error_list.append(np.sum(Point_data[:, 1], axis=0))

        return Point_data[:, 0], centroid, Error_list


def kmeans(image: str) -> None:
    image1 = mpimg.imread(image)
    # Delete the 4th column, Get RGB columns
    RGB = image1[:, :, 0:3]
    # Reshape for calculation
    pixel_vals = RGB.reshape((-1, 3))

    k = [3, 5, 7]

    # The below line of code defines the criteria for the algorithm to stop running,
    # which will happen is 100 iterations are run or the accuracy becomes 95%
    iteration = 100
    accuracy = 0.95

    for i in k:
        print(i)
        kmeans = K_means()
        labels, centroid, Error_list = kmeans.calculation(pixel_vals, i, iteration, accuracy)

        labels = labels.astype(np.int32)
        segmented_data = centroid[labels.flatten()]

        fourth_column = np.ones(segmented_data.shape[0])
        segmented_data = np.column_stack((segmented_data, fourth_column))
        segmented_image = segmented_data.reshape((image1.shape))

        plt.imshow(segmented_image)
        pylab.show()

        plt.plot(Error_list, color='red', marker='o', label="Distortion_measure")
        plt.xlabel('Number of Iterations', fontsize=12)
        plt.ylabel('Sum of Distortion_measure', fontsize=12)
        plt.legend(loc='upper right', fontsize=12)
        plt.show()

if __name__ == "__main__":
    dataset = ("C:/Users/Lucky/PycharmProjects/ml_hw4/umn_csci.png")
    kmeans(dataset)
