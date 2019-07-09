import keras
from keras import datasets
from keras.utils import np_utils
import warnings
warnings.filterwarnings('ignore')

class MNIST_Dataset():
    def __init__(self):
        # Load data from keras.datasets
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

        # Reshape the train / test dataset
        img_rows, img_cols = x_train.shape[1:]
        x_train = x_train.reshape(-1, img_rows, img_cols, 1)
        x_test  = x_test.reshape(-1, img_rows, img_cols, 1)

        # Normalize
        x_train = x_train / 255
        x_test  = x_test  / 255
        self.x_train = x_train.astype('float32')
        self.x_test = x_test.astype('float32')

        # Transform y label to one-hot encoding
        self.y_train = np_utils.to_categorical(y_train)
        self.y_test  = np_utils.to_categorical(y_test)

        self.input_shape = x_train.shape[1:]
