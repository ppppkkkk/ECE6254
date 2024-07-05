import numpy as np
import matplotlib.pyplot as plt


# Define the functions f(x, y) and g(x, y)
def f(x, y):
    return (x - 5) ** 2 + 2 * (y + 3) ** 2 + x * y


def g(x, y):
    return (1 - (y - 3) ** 2) + 10 * ((x + 4) - (y - 3) ** 2) ** 2


# Define the gradient of f(x, y)
def grad_f(x, y):
    df_dx = 2 * (x - 5) + y
    df_dy = 4 * (y + 3) + x
    return np.array([df_dx, df_dy])


# Define the gradient of g(x, y)
def grad_g(x, y):
    dg_dx = 20 * ((x + 4) - (y - 3) ** 2)
    dg_dy = -2 * (1 - (y - 3) ** 2) - 40 * ((x + 4) - (y - 3) ** 2) * (y - 3)
    return np.array([dg_dx, dg_dy])


# Gradient Descent Algorithm
def gradient_descent(grad, initial_point, learning_rate, max_iterations, tolerance):
    point = initial_point
    points_visited = [point]
    values_visited = [f(*point)]

    for _ in range(max_iterations):
        gradient = grad(*point)
        next_point = point - learning_rate * gradient
        points_visited.append(next_point)
        values_visited.append(f(*next_point))

        # Termination condition
        if np.linalg.norm(gradient) < tolerance:
            break

        point = next_point

    return points_visited, values_visited


# Parameters
initial_point_f = np.array([0, 2])
learning_rate_f = 0.01
max_iterations_f = 1000
tolerance_f = 1e-6

# Run Gradient Descent for f(x, y)
points_visited_f, values_visited_f = gradient_descent(
    grad_f, initial_point_f, learning_rate_f, max_iterations_f, tolerance_f
)

# Plot the convergence for f(x, y)
plt.figure(figsize=(14, 7))
plt.plot(values_visited_f, label='f(x, y)')
plt.xlabel('Iteration')
plt.ylabel('Function value')
plt.title('Convergence of Gradient Descent on f(x, y)')
plt.legend()
plt.grid()
plt.show()
