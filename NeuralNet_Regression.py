import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from statistics import mean
from Utils import Utils

class NeuralNetwork():
    # initialise layers
    def __init__(self, lr, layers=[]):
        self.lr= lr
        self.layers = layers 
        self.hidden = []

    # initialise the weights and bias matrix for each layer
    def init_layers(self, inputs):
        for i in range(1,len(inputs)):
            self.layers.append([
                np.random.rand(inputs[i-1], inputs[i])-0.5,
                np.ones((1,inputs[i]))
            ])
        return

    # forward pass through the network
    def forward(self, batch):
        # stores the activation at each layer
        self.hidden = [batch.copy()]
 
        for i in range(len(self.layers)):
            # matrix multiplication between activation and the weights at each layer + the bias
            batch = np.matmul(batch, self.layers[i][0]) + self.layers[i][1]

            # relu activation for each layer, except the last layer
            if i < len(self.layers)-1:
                batch = np.maximum(batch,0)
            self.hidden.append(batch.copy())
        
        # returning the predicted target after each forward pass
        return batch
    
    # root mean sqauare error
    def rmse(self, actual, predicted):
        return np.sqrt(np.mean((predicted-actual)**2))
    
    # gradient of the sum_of_square_error
    def sse_grad(self, actual, predicted):
        return 2*(predicted - actual)
    

    # backward pass through the network
    def backward(self, grad):
        for i in range(len(self.layers)-1, -1, -1):
            
            # compute the gradient w.r.t. the Relu activation function, 
            if i!=len(self.layers)-1:
                grad = np.multiply(grad, np.heaviside(self.hidden[i+1], 0))
            
            # gradient wrt weights
            w_grad = self.hidden[i].T @ grad

            # gradient wrt bias
            b_grad = np.mean(grad, axis=0)
            
            # moving weights and bias in the direction of descent
            self.layers[i][0] = self.layers[i][0] - w_grad*lr
            self.layers[i][1] = self.layers[i][1] - b_grad*lr

            # propagate gradient backwards
            grad = grad @ self.layers[i][0].T
        
        return


def train(x_train, y_train, layer_conf, lr, epochs, size_train, batch_size):

    # instantiate the Neural Network
    network = NeuralNetwork(lr)

    # initialise the layers in the network.
    network.init_layers(layer_conf)

    loss_per_epoch = []

    # iterate through the epochs
    for epoch in range(epochs):
        epoch_loss = []

        # iterate through the batches of the data for each epoch
        for i in range(0, size_train, batch_size):
            x_batch = x_train[i:(i+batch_size)]
            y_batch = y_train[i:(i+batch_size)]
            
            # get the predicted target from the forward pass through the network
            pred = network.forward(x_batch)

            # calculate loss per batch
            loss = network.sse_grad(y_batch, pred)

            # log the root-mean-square-error for each forward pass
            root_mean_square_error = network.rmse(y_batch, pred)
            epoch_loss.append(root_mean_square_error)

            network.backward(loss)
        
        print('Epoch {} Train RMSE : {}'.format(epoch, mean(epoch_loss)))
        loss_per_epoch.append(mean(epoch_loss))

    plt.figure(1)
    plt.plot(range(epochs), loss_per_epoch)
    plt.xlabel('Epoch')
    plt.title('Training Curve')

    return network.layers

# predict using the forward pass and the trained weights and biases.
def test(x_test, y_test, layers, lr):

    network = NeuralNetwork(lr, layers)
    pred_test = network.forward(x_test)
    rmse_test = np.mean(network.rmse(y_test, pred_test))
    print('RMSE Test : {}'.format(rmse_test))

    plt.figure(figsize=(15,8))
    plt.plot(range(1,193), y_test, color='b', linewidth=1, label='label')
    plt.plot(range(1,193), pred_test, color='r', linewidth=1, label='predict')


    plt.xlabel('#th Case')
    plt.ylabel('Heating Load')
    plt.legend()
    plt.show()


if __name__ == '__main__':

    # import data from file
    data = pd.read_csv('../energy_efficiency_data.csv');
    utils = Utils(data)

    # check for missing or null values
    column, isMissingValues = utils.is_missing_values()
    if isMissingValues:
        sys.exit(f'Missing values in {column}')

    # one-hot encode the categorical values columns
    data = utils.one_hot_encode(['Orientation', 'Glazing Area Distribution'])

    # get the training and test set
    data_train, data_test =  utils.split_train_test()

    # split into features and traget
    x_train, y_train = utils.get_X_Y(data_train, 'Heating Load', ['Heating Load', 'Cooling Load'])
    x_test, y_test = utils.get_X_Y(data_test,'Heating Load', ['Heating Load', 'Cooling Load'])

    y_train = y_train.reshape(576,1)
    y_test = y_test.reshape(192,1)

    # layer configurations
    layer_conf = [16, 10, 10, 1]

    # learning rate - can be tuned for better results
    lr = 0.00001

    # Number of epochs - has to be tuned to generalise and avoid overfitting
    epochs = 600

    # number of input instances in a single batch of training.
    batch_size = 32

    # get the weights and biases after training.
    layers = train(x_train, y_train, layer_conf, lr, epochs, x_train.shape[0],batch_size)

    # evaluate the performance of the trained network on test data.
    test(x_test, y_test, layers, lr)


