# Linearly interpolate a series of Bezier curves to form a surface

from bezier_improved import get_bezier_curve
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# 2d
def discretize_curve(curve, ts):
    return np.array([curve.evaluate(t) for t in ts])

def main():
    l_max = 1

    subdivisions = 100
    thetas = np.linspace(0, np.pi, subdivisions)
    curves = []
    for theta in thetas:
        curve = get_bezier_curve(theta, l_max)
        curves.append((theta, curve))
    
    # linearly interpolate the curves to form a surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ts = np.linspace(0, 1, subdivisions)
    points = np.zeros((len(curves), len(ts), 3))
    for i, (theta, curve) in enumerate(curves):
        points[i, :, :2] = discretize_curve(curve, ts).reshape(-1, 2)
        points[i, :, 2] = theta

    ax.plot_surface(points[:, :, 0], points[:, :, 1], points[:, :, 2], cmap='viridis')
    plt.show(block=True)


if __name__ == "__main__":
    main()