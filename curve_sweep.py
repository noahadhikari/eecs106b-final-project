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

    subdivisions = 12
    thetas = np.linspace(0, np.pi, subdivisions)
    curves = []
    for theta in thetas:
        try:
            curve = get_bezier_curve(theta, l_max)
            curves.append((theta, curve))
        except ValueError:
            continue

    # linearly interpolate the curves to form a surface
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)
    # ax.set_aspect('equal', 'box')
    # ax.set_xlim(-0.1 * l_max, l_max)
    # ax.set_ylim(-l_max, l_max)
    # ax.set_zlim(0, np.pi)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # ax.set_xlim(-0.1 * l_max, l_max)
    # ax.set_zlabel('theta (rad)')
    ax.title.set_text('Gripper Bezier Curve Interpolated Surface Top View')
    

    ts = np.linspace(0, 1, len(curves) * 100)
    points = np.zeros((len(curves), len(ts), 3))
    for i, (theta, curve) in enumerate(curves):
        points[i, :, :2] = discretize_curve(curve, ts).reshape(-1, 2)
        points[i, :, 2] = theta

    # ax.plot_surface(points[:, :, 0], points[:, :, 1], points[:, :, 2], cmap='viridis')
    

    # plot the curves only
    for i in range(len(curves)):
        color = plt.cm.viridis(i / len(curves))
        ax.plot(points[i, :, 0], points[i, :, 1], c=color)
        # ax.plot(points[i, :, 0], points[i, :, 1], points[i, :, 2], c=color)
    plt.show(block=True)


if __name__ == "__main__":
    main()