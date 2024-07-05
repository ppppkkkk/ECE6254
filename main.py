import sklearn
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

from torchvision import datasets, transforms


def dataset_load_iris():
    iris_data = sklearn.datasets.load_iris(as_frame=True)
    x = iris_data.data
    y = iris_data.target
    # (i)
    num_examples = x.shape[0]
    print(f"Number of examples in the dataset: {num_examples}")
    # (ii)
    num_features = x.shape[1]
    print(f"Number of features: {num_features}")
    # (iii)
    for feature in iris_data.data.columns:
        print(f"\nFeature: {feature}")
        print(f"Min value: {iris_data.data[feature].min()}")
        print(f"Max value: {iris_data.data[feature].max()}")
        print(f"Mean value: {iris_data.data[feature].mean()}")
        print(f"Standard Deviation: {iris_data.data[feature].std()}")
    # (iv)
    class_counts = np.bincount(y)
    total_examples = len(y)
    class_percentages = class_counts / total_examples * 100
    for idx, percentage in enumerate(class_percentages):
        print(f"Percentage of examples for class {idx}: {percentage:.2f}%")

    if np.all(class_counts == class_counts[0]):
        print("\nThe dataset is balanced.")
    else:
        print("\nThe dataset is imbalanced.")


def plot_two_feature():
    import matplotlib.pyplot as plt
    iris_data = sklearn.datasets.load_iris()
    x = iris_data.data
    y = iris_data.target
    feature1 = x[:, 0]
    feature2 = x[:, 1]
    plt.figure(figsize=(10, 6))
    # Plot each class with a different color
    for class_num in np.unique(y):
        idx = np.where(y == class_num)
        plt.scatter(feature1[idx], feature2[idx], label=iris_data.target_names[class_num])

    plt.xlabel(iris_data.feature_names[0])
    plt.ylabel(iris_data.feature_names[1])
    plt.title("")
    plt.legend()
    plt.show()


def dataset_load_CIFAR():
    import torch
    import torchvision
    transform = transforms.Compose([transforms.ToTensor()])

    # load CIFAR-10
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Combine training and test set
    data = torch.utils.data.ConcatDataset([train_set, test_set])

    # data to images and labels
    images, labels = zip(*data)
    labels = torch.tensor(labels)

    # (i)
    num_examples = len(images)
    unique_labels = torch.unique(labels)
    print(f"Number of examples in the dataset: {num_examples}")
    print(f"Values of class labels: {unique_labels.numpy()}")

    # (ii)
    class_counts = torch.bincount(labels)
    total_examples = len(labels)
    class_percentages = class_counts / total_examples * 100

    for idx, percentage in enumerate(class_percentages):
        print(f"Percentage of examples for class {idx}: {percentage:.2f}%")

    if torch.all(class_counts == class_counts[0]):
        print("\nThe dataset is balanced.")
    else:
        print("\nThe dataset is imbalanced.")


if __name__ == '__main__':
    plot_two_feature()
    #dataset_load_CIFAR()
