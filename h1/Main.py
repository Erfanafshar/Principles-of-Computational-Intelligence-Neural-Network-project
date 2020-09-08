import numpy as np
import csv
from itertools import islice
import matplotlib.pyplot as plt

number_of_data = 180
number_of_train_data = 150
number_of_test_data = 30


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def set_date():
    tempX = []
    y0 = []
    with open('dataset.csv', 'r') as dataset:
        dataset_reader = csv.reader(dataset)
        for row in islice(dataset_reader, 1, None):
            tempX.append([float(row[0]), float(row[1])])
            y0.append(float(row[2]))
        X = np.array(tempX)
    return X, y0


def plot(start_index, end_index, colors_array, bias):
    for i in range(start_index, end_index):
        if colors_array[i - bias] == 1.0:
            plt.scatter(X[i][0], X[i][1], 2, "red")
        if colors_array[i - bias] == 0.0:
            plt.scatter(X[i][0], X[i][1], 2, "blue")

    plt.title('Dataset')
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.legend(["0", "1"])
    plt.show()


def initialize_values():
    gradW = np.array([0.0, 0.0])
    W = np.array([np.random.rand(), np.random.rand()])
    b = np.random.rand()
    gradB = 0
    n_epoch = 1000
    lr = 3.0
    return gradW, W, b, gradB, n_epoch, lr


def train():
    global b
    global W
    for i in range(n_epoch):
        gradW[0] = 0
        gradW[1] = 0
        gradB = 0
        total_cost = 0
        for j in range(number_of_train_data):
            u = np.dot(W, X[j]) + b
            y = sigmoid(u)
            this_cost = -1 * (y0[j] * np.log(y) + ((1 - y0[j]) * np.log(1 - y)))
            total_cost += this_cost

            dcost_dy = -1 * ((y0[j] / y) + (-1 * (1 - y0[j]) / (1.0 - y)))

            dy_dw0 = (X[j][0]) * sigmoid_derivative(sigmoid(u))
            gradW[0] += dcost_dy * dy_dw0

            dy_dw1 = (X[j][1]) * sigmoid_derivative(sigmoid(u))
            gradW[1] += dcost_dy * dy_dw1

            dy_db = sigmoid_derivative(sigmoid(u))
            gradB += dcost_dy * dy_db

        W[0] -= lr * gradW[0] / number_of_train_data
        W[1] -= lr * gradW[1] / number_of_train_data
        b -= lr * gradB / number_of_train_data
        print(total_cost)
    # print(W[0])
    # print(W[1])
    # print(gradB)


def test():
    global W
    correct_number = 0
    predicted_label = []
    for i in range(150, 180):
        u = np.dot(W, X[i]) + b
        y = sigmoid(u)
        if y >= 0.5:
            predicted_label.append(1)
            if y0[i] == 1.0:
                correct_number += 1

        if y < 0.5:
            predicted_label.append(0)
            if y0[i] == 0.0:
                correct_number += 1

    accuracy = correct_number / 30
    print("accuracy : ", accuracy)
    return predicted_label


X, y0 = set_date()
gradW, W, b, gradB, n_epoch, lr = initialize_values()
train()
predicted_label = test()

#plot(0, number_of_data, y0, 0)
#plot(number_of_train_data, number_of_data, predicted_label, number_of_train_data)
#plot(number_of_train_data, number_of_data, y0, 0)
