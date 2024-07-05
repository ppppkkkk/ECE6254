import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from skorch import NeuralNetClassifier
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skorch.helper import predefined_split


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

X_train = trainset.data
y_train = torch.tensor(trainset.targets)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
X_test = testset.data
y_test = torch.tensor(testset.targets)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")


class SimpleNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleNN, self).__init__()

        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


valid_ds = torch.utils.data.TensorDataset(torch.tensor(X_val.astype('float32')), y_val)

net = NeuralNetClassifier(
    SimpleNN,
    criterion=nn.CrossEntropyLoss,
    optimizer=optim.Adam,
    lr=0.001,
    max_epochs=20,
    batch_size=64,
    train_split=None,
    iterator_train__shuffle=True,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

net.fit(X_train.astype('float32'), y_train)
train_loss = [epoch['train_loss'] for epoch in net.history]

'''
plt.plot(train_loss, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.legend()
plt.show()
'''

from sklearn.model_selection import GridSearchCV

param_grid = {
    'lr': [0.001, 0.005, 0.01, 0.05],
    'batch_size': [32, 64]
}

gs = GridSearchCV(net, param_grid, refit=False, cv=3, scoring='accuracy', verbose=2)

gs.fit(X_train.astype('float32'), y_train)
print("Best parameters found: ", gs.best_params_)


best_params = gs.best_params_

net.set_params(lr=best_params['lr'], batch_size=best_params['batch_size'])
net.fit(X_train.astype('float32'), y_train)

accuracy = net.score(X_test.astype('float32'), y_test)
print(f"Test accuracy: {accuracy*100:.2f}%")