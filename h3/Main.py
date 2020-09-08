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
    W = np.array([np.random.rand(), np.random.rand()])
    V = np.array([np.random.rand(), np.random.rand()])
    U = np.array([np.random.rand(), np.random.rand()])
    b0 = np.random.rand()
    b1 = np.random.rand()
    b2 = np.random.rand()
    gradW = np.array([0.0, 0.0])
    gradV = np.array([0.0, 0.0])
    gradU = np.array([0.0, 0.0])

    Z = np.array([0.0, 0.0])
    n_epoch = 5000
    lr = 0.06
    return W, V, U, b0, b1, b2, gradW, gradV, gradU, Z, n_epoch, lr


def train():
    global b0
    global b1
    global b2
    global W
    global V
    global U
    for i in range(n_epoch):
        for j in range(number_of_train_data):
            k = np.dot(W, X[j]) + b0
            Z[0] = sigmoid(k)
            h = np.dot(V, X[j]) + b1
            Z[1] = sigmoid(h)
            t = np.dot(U, Z) + b2
            y = sigmoid(t)

            this_cost = pow((y - y0[j]), 2)

            dcost_dy = 2 * (y - y0[j])

            dy_du0 = sigmoid_derivative(sigmoid(t)) * Z[0]
            gradU[0] = dcost_dy * dy_du0

            dy_du1 = sigmoid_derivative(sigmoid(t)) * Z[1]
            gradU[1] = dcost_dy * dy_du1

            dy_db2 = sigmoid_derivative(sigmoid(t))
            gradB2 = dcost_dy * dy_db2

            dcost_dz0 = 2 * (y - y0[j]) * sigmoid_derivative(sigmoid(t)) * U[0]

            dz0_dw0 = sigmoid_derivative(sigmoid(k)) * X[j][0]
            gradW[0] = dcost_dz0 * dz0_dw0

            dz0_dw1 = sigmoid_derivative(sigmoid(k)) * X[j][1]
            gradW[1] = dcost_dz0 * dz0_dw1

            dz0_db0 = sigmoid_derivative(sigmoid(k))
            gradB0 = dcost_dz0 * dz0_db0

            dcost_dz1 = 2 * (y - y0[j]) * sigmoid_derivative(sigmoid(t)) * U[1]

            dz1_dv0 = sigmoid_derivative(sigmoid(h)) * X[j][0]
            gradV[0] = dcost_dz1 * dz1_dv0

            dz1_dv1 = sigmoid_derivative(sigmoid(h)) * X[j][1]
            gradV[1] = dcost_dz1 * dz1_dv1

            dz1_db1 = sigmoid_derivative(sigmoid(h))
            gradB1 = dcost_dz1 * dz1_db1

            W[0] -= lr * gradW[0]
            W[1] -= lr * gradW[1]
            b0 -= lr * gradB0

            V[0] -= lr * gradV[0]
            V[1] -= lr * gradV[1]
            b1 -= lr * gradB1

            U[0] -= lr * gradU[0]
            U[1] -= lr * gradU[1]
            b2 -= lr * gradB2
            print(this_cost)


def test():
    global W
    global V
    global U
    correct_number = 0
    predicted_label = []
    for i in range(150, 180):
        k = np.dot(W, X[i]) + b0
        Z[0] = sigmoid(k)
        h = np.dot(V, X[i]) + b1
        Z[1] = sigmoid(h)
        t = np.dot(U, Z) + b2
        y = sigmoid(t)

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
W, V, U, b0, b1, b2, gradW, gradV, gradU, Z, n_epoch, lr = initialize_values()
train()
predicted_label = test()

#plot(0, number_of_data, y0, 0)
#plot(number_of_train_data, number_of_data, predicted_label, number_of_train_data)
#plot(number_of_train_data, number_of_data, y0, 0)
