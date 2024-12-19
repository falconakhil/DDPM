import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_dataloader(batch_size=64):
    transform = transforms.Compose(
        [
            transforms.Resize(size=(32,32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2*x-1),
            # transforms.Normalize(mean=cifar10_means,std=cifar10_std)
        ]
    )


    train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True,transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True,transform=transform
    )

    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

    return train_loader, test_loader
