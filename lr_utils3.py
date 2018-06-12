from sklearn.datasets import fetch_mldata
import numpy as np

def load_dataset():
    mnist = fetch_mldata('MNIST original')
    # rescale the data, use the traditional train/test split
    X, y = mnist.data / 255., mnist.target
   
    train_set_x_orig = X[:60000] # your train set features
    
    train_set_y = y[:60000] # your train set labels
    train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))
    train_set_y_orig = np.empty_like([])
    for i in range(train_set_y.shape[1]):
        if(train_set_y[0,i]>=5):
            train_set_y_orig= np.append(train_set_y_orig,[1])
        else:
            train_set_y_orig= np.append(train_set_y_orig,[0])
        
        
    test_set_x_orig = X[60000:] # your test set features
    test_set_y = y[60000:] # your test set labels
    
    test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))
    test_set_y_orig = np.empty_like([])
    for i in range(test_set_y.shape[1]):
        if(test_set_y[0][i]>=5):
            test_set_y_orig= np.append(test_set_y_orig,[1])
        else:
            test_set_y_orig= np.append(test_set_y_orig,[0])
        #print test_set_y_orig[i]
    
    classes = np.array(['less than 5','greater than equal to 5']) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
   
    return train_set_x_orig, train_set_y_orig.astype(int), test_set_x_orig, test_set_y_orig.astype(int), classes

load_dataset()