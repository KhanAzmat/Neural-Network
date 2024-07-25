import numpy as np
import pandas as pd


class Utils:
    def  __init__(self, data):
        self.data = data

    # to check for missing values
    def is_missing_values(self):
        for index,value in self.data.isnull().sum().items():
            if value > 0:
                return (index, True)
        return (None, False)
    
    # to one-hot-encode the categorical features
    def one_hot_encode(self, columns):
        self.data = pd.get_dummies(self.data, columns=['Orientation', 'Glazing Area Distribution'], dtype=float)
        return self.data
    
    # to drop a feature
    def drop_col(self, col):
        self.data.drop(columns=(col), inplace=True)
        return self.data
        
    # split the data frame into training and testing data
    def split_train_test(self):
        # 75% of data is training data and the remaining is test data.
        train_size = int(0.75*len(self.data))

        #training and test set 
        train_data = self.data[:train_size]
        test_data = self.data[train_size:]

        return train_data, test_data
    
    # return features and target
    def get_X_Y(self,df, colY, col=None):
        # dropping target and converting to numpy array
        X = df.drop(col, axis=1).to_numpy()

        # normalising each feature
        X = (X-np.min(X, axis=0))/(np.max(X,axis=0)-np.min(X,axis=0))

        # get target
        Y = df[colY].to_numpy()

        return X, Y

