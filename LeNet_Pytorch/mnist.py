from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')

def MNIST_dataset(batch_size):
    trans_img = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_set = MNIST('./data', train=True, download=True, transform=trans_img)
    test_set = MNIST('./data', train=False, download=True, transform=trans_img)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader
