import numpy as np
import matplotlib.pyplot as plt


def generate_data(dim, num_points=50):
    return np.random.uniform(-0.5, 0.5, (num_points, dim))


def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)


def angle_between(point1, point2):
    dot_product = np.dot(point1, point2)
    norms = np.linalg.norm(point1) * np.linalg.norm(point2)
    return np.arccos(dot_product / norms)


def compute_distances_angles(data):
    num_points = data.shape[0]
    distances = []
    angles = []

    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                dist = euclidean_distance(data[i], data[j])
                angle = angle_between(data[i], data[j])
                distances.append(dist)
                angles.append(angle)

    return distances, angles


def plot_scatter(distances, angles, title):
    plt.scatter(distances, angles, alpha=0.6)
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Angle (in radians)")
    plt.title(title)
    plt.show()


# Part (a)
data_5d = generate_data(5)
distances_5d, angles_5d = compute_distances_angles(data_5d)
plot_scatter(distances_5d, angles_5d, "Distance vs Angle for 5D data")

# Part (b)
data_50d = generate_data(50)
distances_50d, angles_50d = compute_distances_angles(data_50d)
plot_scatter(distances_50d, angles_50d, "Distance vs Angle for 50D data")

# Part (c)
data_100d = generate_data(100)
distances_100d, angles_100d = compute_distances_angles(data_100d)
plot_scatter(distances_100d, angles_100d, "Distance vs Angle for 100D data")