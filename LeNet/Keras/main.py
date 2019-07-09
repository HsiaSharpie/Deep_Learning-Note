import keras
from mnist import MNIST_Dataset
from lenet import LeNet
import matplotlib.pyplot as plt

# plot loss
def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.legend(['train', 'validation'], loc='upper right')

def main():
    # hyper-parameters
    epochs = 10
    batch_size = 64

    # build LeNet model
    mnist_data = MNIST_Dataset()
    LeNet_model = LeNet.build_model(mnist_data.input_shape, classes=10)
    LeNet_model.summary()
    LeNet_model.compile(loss=keras.losses.categorical_crossentropy, optimizer='SGD')

    # train & test the LeNet model
    history = LeNet_model.fit(mnist_data.x_train, mnist_data.y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    score = LeNet_model.evaluate(mnist_data.y_train, mnist_data.y_test, batch_size=batch_size)
    print('Test Score: {}'.format(score))
    
    plot_loss(history)
    plt.show()

if __name__ == '__main__':
    main()
