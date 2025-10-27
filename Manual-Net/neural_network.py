import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv')
data = np.array(data)
rows, cols = data.shape
print(f"There are {rows} rows and {cols} columns")
np.random.shuffle(data)

# Development set
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1: cols]
X_dev = X_dev / 255.

# Training set
data_train = data[1000: rows].T
Y_train = data_train[0]
X_train = data_train[1: cols]
X_train = X_train / 255.
_, m_train = X_train.shape

def init_prams():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    A = np.exp(Z - np.max(Z, axis=0)) / sum(np.exp(Z - np.max(Z, axis=0)))
    return A

def deriv_ReLU(Z):
    return Z > 0

def swish(Z, B):
    return Z * 1/(1-np.exp(-B*Z))

def derSwish(Z, B):
    return Z*B > 0.5
    

def froward_prop(W1, b1, W2, b2, X):
    Z1 =  W1.dot(X) + b1
    A1 = swish(Z1, 10)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.max()+1, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * derSwish(Z1, 10)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_prediction(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions==Y)/Y.size

def gradient_descent(X, Y, interations, alpha):
    W1, b1, W2, b2 = init_prams()
    # To store accuracy for plotting
    iterations_list = []
    accuracy_list = []
    for i in range(interations):
        Z1, A1, Z2, A2 = froward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:
            print("Iterations: ", i)
            predictions = get_prediction(A2)
            accuracy = get_accuracy(predictions, Y)
            print("Accuracy: ", accuracy)
            iterations_list.append(i)
            accuracy_list.append(accuracy)

    return W1, b1, W2, b2, iterations_list, accuracy_list

# --- New Plotting Functions ---

def plot_accuracy(iterations, accuracies):
    """
    Plots the training accuracy over iterations.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, accuracies, label='Training Accuracy')
    plt.title('Training Accuracy over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_weights(W1):
    """
    Visualizes the weights of the first layer as images.
    """
    W1 = W1.reshape(10, 28, 28)
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    for i in range(10):
        axes[i].imshow(W1[i], cmap='gray')
        axes[i].set_title(f"Neuron {i+1}")
        axes[i].axis('off')
    plt.suptitle('First Layer Learned Weights')
    plt.show()

# --- Run Training and Visualize ---

W1, b1, W2, b2, iterations, accuracies = gradient_descent(X_train, Y_train, 1000, 0.1)

# Plot the results
plot_accuracy(iterations, accuracies)
plot_weights(W1)