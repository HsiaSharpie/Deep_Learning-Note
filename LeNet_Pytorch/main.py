from lenet import Lenet
from mnist import MNIST_dataset
import torch
import torch.nn as nn
import torch.optim as optim

def train_and_test(device, epoches, train_loader, test_loader, optimizer, criterion, model):
    iteration = 0
    training_loss = {}
    for epoch in range(1, epoches):
        for index, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            training_loss[iteration] = loss.item()

            iteration += 1
            if iteration % 600 == 0:
                print('Training Loss in Epoch:{}, loss: {}'.format(epoch, loss.item()))

                correct = 0
                total = 0
                for index, (images, targets) in enumerate(test_loader):
                    images, targets = images.to(device), targets.to(device)

                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)

                    total += targets.size(0)
                    correct += (predicted == targets).numpy().sum()

                accuracy = 100 * (correct / total)
                print('Testing Accuracy in Epoch:{}, Accuracy: {}'.format(epoch, accuracy))
                print('-'*50)

    return training_loss

def plot_loss(loss):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(list(loss.keys()), list(loss.values()))
    plt.show()


def main():
    # Set the hyper-parameters
    learning_rate = 0.001
    batch_size = 100
    epoches = 11

    train_loader, test_loader = MNIST_dataset(batch_size)

    # Build the model
    lenet = Lenet()
    # Set the loss function & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(lenet.parameters(), lr=learning_rate)

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Train & Test the model
    training_loss = train_and_test(device, epoches, train_loader, test_loader, optimizer, criterion, model=lenet)
    plot_loss(training_loss)


if __name__ == '__main__':
    main()
