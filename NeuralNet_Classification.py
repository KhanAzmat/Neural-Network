import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from statistics import mean
from Utils import Utils
import math

# implementation of the RelU function
class Relu():
    def __init__(self):
        pass

    def forward(self, x):
        return np.maximum(x, 0)

    def backward(self, grad, lr, prev_hidden):
        return np.multiply(grad, np.heaviside(prev_hidden, 0))


# to implement each layer
class Dense():
    def __init__(self, input_size, output_size, bias=True, activation=True, seed=0):
        self.add_bias = bias
        self.add_activation = activation
        self.hidden = None
        self.prev_hidden = None

        # set seed for reproducible results
        np.random.seed(seed)

        k = math.sqrt(1 / input_size)

        # initailise weights matrix with random weights
        self.weights = np.random.rand(input_size, output_size) * (2 * k) - k

        # initialise the bias randomly
        self.bias = np.ones((1, output_size)) * (2 * k) - k\
        
        # using relu activation, except the last layer
        self.activation = Relu()

        super().__init__()

    # forward pass through each layer
    def forward(self, x):
        self.prev_hidden = x.copy()
        # matrix multiplication between inputs to the layer and weights
        x = np.matmul(x, self.weights)

        # add bias if required
        if self.add_bias:
            x += self.bias

        # use Relu as activation, except for the last layer
        if self.add_activation:
            x = self.activation.forward(x)
        self.hidden = x.copy()

        # return input to the next layer/prediction
        return x

    def backward(self, grad, lr):

        # if Relu activation for the current layer, then derive the gradient of the loss wrt the Relu function first.
        if self.add_activation:
            grad = self.activation.backward(grad, lr, self.hidden)

        # gradient wrt the weights
        w_grad = self.prev_hidden.T @ grad

        # gradient wrt the bias
        b_grad = np.mean(grad, axis=0)

        # tweak the weights in the direction of negative gradient
        self.weights -= w_grad * lr

        # adjust bias in the direction of negative gradient
        if self.add_bias:
            self.bias -= b_grad * lr

        # update gradient for previous layer
        grad = grad @ self.weights.T
        return grad

class ClassificationNet():
    def __init__(self, output_size=1):
        # Setup 3 neural network layers.  Each Dense class is a single network layer.
        # We don't use relu activation on the last layer, but we do in the previous layers.
        self.layer1 = Dense(input_size=33, output_size=25)
        self.layer2 = Dense(input_size=25, output_size=25)
        self.layer3= Dense(input_size=25, output_size=output_size, activation=False)


    def forward(self, x):
        # In the forward pass, we take in input data, and run our 2 layers over the data.
        x = self.layer1.forward(x)
        x_latent = self.layer2.forward(x)
        pred = self.layer3.forward(x_latent)
        return pred, x_latent

    def backward(self, grad, lr):
        # In the backward pass, we take the gradient and learning rate, and use them to adjust parameters in each layer.
        grad = self.layer3.backward(grad, lr)
        grad = self.layer2.backward(grad, lr)
        self.layer1.backward(grad, lr)

# activation for binary classifiaction[0,1]
sigmoid = lambda x : 1/(1+ np.exp(-x))

#Cross-entropy loss, neagtive log-likelihood
# a small arbitrary value added to avoid taking log of 0.
tol = 1e-6
nll = lambda pred, actual : -(actual * np.log(pred + tol) + (1-actual) * np.log(1-pred+tol))

# gradient of the loss
nll_grad = lambda pred, actual : pred - actual

def train(lr, epochs, x_train, y_train, x_test, y_test):
    net = ClassificationNet(1)

    loss_per_epoch = []

    for epoch in range(epochs):
        epoch_loss = 0
        
        for x, target in zip(x_train, y_train):
            # Run the sigmoid function over the output of the neural network
            pred, x_latent = net.forward(x.reshape(1,-1))
            pred = sigmoid(pred)

            # Compute the gradient using the nll_grad function
            grad = nll_grad(pred, target)
            epoch_loss += nll(pred, target)[0,0]

            # Update the network parameters
            net.backward(grad, lr)
            
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} train loss: {epoch_loss / len(x_train)}")
            
            loss_per_epoch.append(epoch_loss/len(x_train))
            epoch_loss = 0
            pred_y = []

            # evaluate the model after each epoch of training
            for x, target in zip(x_test, y_test):
                pred, x_latent = net.forward(x.reshape(1,-1))
                pred = sigmoid(pred)
                pred_y.append(pred)
                epoch_loss += nll(pred, target)[0,0]
            print(f"Valid loss: {epoch_loss / len(x_test)}")
            
    print(np.c_[np.array(pred_y).reshape(88,1), y_test])
    plt.figure()
    plt.plot(range(20), loss_per_epoch)
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.show()



if __name__ == '__main__':

    # import data from file
    data = pd.read_csv('../ionosphere_data.csv', header=None)
    utils = Utils(data)

    # convert class attribute to numeric
    data = data.replace({'b':1, 'g':0}, inplace=True)

    # drop attribute 1, as it does not add any information.
    data = utils.drop_col(1)

    print(data.head())

    # check for missing or null values
    column, isMissingValues = utils.is_missing_values()
    if isMissingValues:
        sys.exit(f'Missing values in {column}')


    # get the training and test set
    data_train, data_test =  utils.split_train_test()

    # split into features and traget
    x_train, y_train = utils.get_X_Y(data_train, 34, [34])
    x_test, y_test = utils.get_X_Y(data_test,34, [34])

    y_train = y_train.reshape(263,1)
    y_test = y_test.reshape(88,1)


    # learning rate
    lr = 0.001

    # number of epochs
    epochs = 200

    # call the training loop
    train(lr, epochs, x_train, y_train, x_test, y_test)


