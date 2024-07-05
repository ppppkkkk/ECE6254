import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from skorch import NeuralNetClassifier
from torch.optim import RMSprop
# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor()])
cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Split data into train-validation-test (60% - 20% - 20%)
train_data, valid_data = train_test_split(cifar_train, test_size=0.2, random_state=42)
test_data = cifar_test

print(f"Train data size: {len(train_data)}")
print(f"Validation data size: {len(valid_data)}")
print(f"Test data size: {len(test_data)}")

# Building the CNN Model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=2**3, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=2**3, out_channels=2**6, kernel_size=5, stride=1, padding=2)
        self.fc = nn.Linear(2**6 * 8 * 8, num_classes)

    def forward(self, x):
        x = F.gelu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.gelu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.softmax(self.fc(x), dim=1)
        return x


X_train = np.array([x[0].numpy() for x in train_data])
y_train = np.array([x[1] for x in train_data], dtype=np.int64)
X_valid = np.array([x[0].numpy() for x in valid_data])
y_valid = np.array([x[1] for x in valid_data], dtype=np.int64)
from skorch.dataset import Dataset
valid_ds = Dataset(X_valid, y_valid)
from skorch.helper import predefined_split
net = NeuralNetClassifier(
    module=lambda: CNN(num_classes=10),
    criterion=nn.CrossEntropyLoss,
    optimizer=RMSprop,
    lr=0.00001, # Initial learning rate
    batch_size=64,
    max_epochs=10,
    iterator_train__shuffle=True,
    device='cuda' if torch.cuda.is_available() else 'cpu',  # Use CUDA if available
    train_split=predefined_split(valid_ds)
)



# Train the model
net.fit(X_train, y_train)

X_test = np.array([x[0].numpy() for x in test_data])
y_test = np.array([x[1] for x in test_data], dtype=np.int64)
test_accuracy = net.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}%")
#b
import matplotlib.pyplot as plt
train_loss = net.history[:, 'train_loss']
valid_loss = net.history[:, 'valid_loss']

plt.plot(train_loss, '-o', label='Training Loss', color='blue')
plt.plot(valid_loss, '-o', label='Validation Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Over Epochs')
plt.show()

#c
lrs = [0.001, 0.002, 0.003]
batch_sizes = [8, 16, 32, 64]

best_valid_loss = float('inf')
best_params = None

plt.figure(figsize=(12, 8))

for lr in lrs:
    for batch_size in batch_sizes:
        # 设置超参数
        net.set_params(lr=lr, batch_size=batch_size, max_epochs=20)
        net.fit(X_train, y_train)

        valid_loss = net.history[:, 'valid_loss']

        plt.plot(valid_loss, '-o', label=f'lr={lr}, batch_size={batch_size}')

        # 记录最佳的超参数和损失值
        current_valid_loss = net.history[-1, 'valid_loss']
        if current_valid_loss < best_valid_loss:
            best_valid_loss = current_valid_loss
            best_params = {
                'lr': lr,
                'batch_size': batch_size
            }

plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Loss Across Different Hyperparameters')
plt.legend()
plt.show()
print(f"Best parameters: {best_params}")

#d Get accuracy on test set
test_accuracy = net.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}%")