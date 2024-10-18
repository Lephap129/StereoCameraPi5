import numpy as np
from scipy.ndimage import sobel

def compute_sad(I1, I2, W, x, y, d):
    """Compute the SAD cost for a given window."""
    sad = 0
    for i, j in W:
        sad += abs(I1[x+i, y+j] - I2[x+i, y+j-d])
    return sad

def compute_grad(I1, I2, W, x, y, d):
    """Compute the gradient-based cost."""
    grad_cost = 0
    # Compute gradients
    grad_x1 = sobel(I1, axis=1)
    grad_x2 = sobel(I2, axis=1)
    grad_y1 = sobel(I1, axis=0)
    grad_y2 = sobel(I2, axis=0)
    for i, j in W:
        grad_cost += abs(grad_x1[x+i, y+j] - grad_x2[x+i, y+j-d]) + abs(grad_y1[x+i, y+j] - grad_y2[x+i, y+j-d])
    return grad_cost

def compute_combined_cost(I1, I2, W, x, y, d, lambd):
    """Compute the combined cost using SAD and gradient measures."""
    C_SAD = compute_sad(I1, I2, W, x, y, d)
    C_GRAD = compute_grad(I1, I2, W, x, y, d)
    return lambd * C_SAD + (1 - lambd) * C_GRAD

# Example usage
I1 = np.random.random((100, 100))  # Dummy image 1
I2 = np.random.random((100, 100))  # Dummy image 2

# Define a window W around (x,y)
W = [(i, j) for i in range(-1, 2) for j in range(-1, 2)]  # 3x3 window

x, y, d = 50, 50, 2  # example coordinates and disparity
lambd = 0.5  # example lambda value

cost = compute_combined_cost(I1, I2, W, x, y, d, lambd)
print(f"Combined cost at ({x}, {y}) with disparity {d}: {cost}")
