import torch
from torchvision import datasets, transforms

def get_mnist_data(data_path, batch_size):
    dataset_transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])

    train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=dataset_transform)
    test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=dataset_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader
