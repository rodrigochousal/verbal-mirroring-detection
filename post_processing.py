import matplotlib.pyplot as plt
import numpy as np

def plot_scatter(matrix):
    '''
    Plot a scatter plot with values y being float and x being time.

    Args:
    matrix (numpy.ndarray): A 2D float matrix.

    Returns:
    None
    '''
    n, m = matrix.shape
    x = np.arange(m)
    y = matrix.flatten()
    plt.scatter(x.repeat(n), y)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()