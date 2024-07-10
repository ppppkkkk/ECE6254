import numpy as np
from sklearn.datasets import load_iris


# Use Iris dataset as an example
def custom_dataset_split(data, labels, ratios):
    """
    Splits the dataset into train, validation, and test sets according to the given ratios.

    Args:
    - data (array-like): The dataset features.
    - labels (array-like): The dataset labels.
    - ratios (list of int): Ratios for [train, validation, test]. Should sum up to 100.

    Returns:
    - splits: Dictionary containing train_data, val_data, test_data, train_labels, val_labels, test_labels.
    """

    # Ensure that the sum of ratios equals 100%
    assert sum(ratios) == 100, "The ratios must sum up to 100%"

    # Compute dataset lengths for each subset
    total_length = len(data)
    train_length = int(total_length * ratios[0] / 100)
    val_length = int(total_length * ratios[1] / 100)
    test_length = total_length - train_length - val_length

    # Shuffling data
    indices = np.arange(total_length)
    np.random.shuffle(indices)

    train_indices = indices[:train_length]
    print(train_indices)
    val_indices = indices[train_length:train_length + val_length]
    test_indices = indices[train_length + val_length:]
    # Splitting data and labels
    splits = {
        "train_data": data[train_indices],
        "val_data": data[val_indices],
        "test_data": data[test_indices],
        "train_labels": labels[train_indices],
        "val_labels": labels[val_indices],
        "test_labels": labels[test_indices]
    }

    return splits


# Testing the function using Iris dataset
if __name__ == "__main__":
    # Loading Iris dataset
    iris = load_iris()
    data = iris.data
    labels = iris.target

    ratios = [60, 30, 10]
    result = custom_dataset_split(data, labels, ratios)
    # Printing the lengths of the splits
    # print(result['train_labels'])
    print(f"Training data length: {len(result['train_data'])}")
    print(f"Validation data length: {len(result['val_data'])}")
    print(f"Test data length: {len(result['test_data'])}")
