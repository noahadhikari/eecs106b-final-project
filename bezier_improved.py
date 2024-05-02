import numpy as np
from bezier.curve import Curve

from mystic.symbolic import generate_constraint, generate_solvers
from mystic.solvers import diffev2

import matplotlib.pyplot as plt



def find_angle(p0, p1, p2):
    v0 = p0 - p1
    v1 = p2 - p1

    if np.linalg.norm(v0) < 1e-6 or np.linalg.norm(v1) < 1e-6:
        return 0

    return np.arccos(v0.T @ v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))

def get_cubic_bezier(params, theta):
    a, c0, c1 = params

    p0 = np.array([0, 0])
    v0 = c0 * np.array([0, -1])
    v1 = c1 * np.array([np.cos(theta), np.sin(theta)])
    p1 = a * np.array([np.sin(theta), -np.cos(theta)])

    # set the control points to be 1 unit of time in velocity vector from the endpoint

    pc0 = p0 + v0
    pc1 = p1 - v1

    # define the curve
    nodes = np.asfortranarray([p0, pc0, pc1, p1]).T
    curve = Curve(nodes, degree=3)

    return curve

def get_quadratic_bezier_cp(params, theta):
    a, c0, _ = params

    p0 = np.array([0, 0])
    p1 = a * np.array([np.sin(theta), -np.cos(theta)])

    # optimize over the control point directly
    pc = np.array([0, -c0])

    # define the curve
    nodes = np.asfortranarray([p0, pc, p1]).T
    curve = Curve(nodes, degree=2)

    return curve

def get_quadratic_bezier_right_angle(params, theta):
    a, _, _ = params

    p0 = np.array([0, 0])
    p1 = a * np.array([np.sin(theta), -np.cos(theta)])
    v0 = np.array([0, -1])
    v1 = np.array([np.cos(theta), np.sin(theta)])

    # find the intersection between the two tangent lines
    # f0(t) = p0 + tv0
    # f1(s) = p1 + sv1
    # f0(t) = f1(s) => p0 + tv0 = p1 + sv1
    # => p0 - p1 = sv1 - tv0
    # => p0 - p1 = [v1, -v0] [s, t]^T
    # => [v1, -v0]^-1 [p0 - p1] = [s, t]^T

    A = np.array([v1, -v0]).T
    if np.linalg.det(A) < 1e-6:
        raise ValueError("Cannot find intersection point.")
    s, t = np.linalg.inv(A) @ (p0 - p1)

    pc = p0 + t * v0

    # define the curve
    nodes = np.asfortranarray([p0, pc, p1]).T
    curve = Curve(nodes, degree=2)

    return curve

def objective(params, theta, l, bezier_fn):
    def obj(curve):
        p0 = curve.nodes[:, 0].reshape(2, 1)
        p1 = curve.nodes[:, -1].reshape(2, 1)
        p2 = curve.nodes[:, -2].reshape(2, 1)
        angle = find_angle(p0, p1, p2)
        # angle calc is unnecessary now, but was useful for the bezier cp version
        return 1 / l * (curve.length - l) ** 2 + (np.sin(np.pi/2) - np.sin(angle)) ** 2 + (np.cos(np.pi/2) - np.cos(angle)) ** 2
    
    return obj(bezier_fn(params, theta))

def plot_curve(params, theta, l_max, bezier_fn, ax=None):
    curve = bezier_fn(params, theta)
    if ax is None:
        ax = plt.subplot(111)
    plt.xlim(-0.05, l_max)
    plt.ylim(-l_max, 0.05)

    # make the plot square
    ax.set_aspect('equal', 'box')


    # draw vertical and angle lines
    vert_line = np.array([[0, 0], [0, -l_max]])
    angle_line = l_max * np.array([[0, 0], [np.sin(theta), -np.cos(theta)]])
    plt.plot(vert_line[:, 0], vert_line[:, 1], 'k-')
    plt.plot(angle_line[:, 0], angle_line[:, 1], 'k-')

    # plot bezier control points, and lines from endpoints to control point

    # connect the dots
    for i in range(len(curve.nodes[0]) - 1):
        plt.plot(curve.nodes[0, i:i + 2], curve.nodes[1, i:i + 2], 'ro--')

    print(f"Curve length for {bezier_fn.__name__}: ", curve.length)
    curve.plot(num_pts=100, ax=ax)

def main():

    # f : R -> R^2, t in [0, 1] -> (x(t), y(t)) given by a quadratic bezier curve
    # endpoints are p0 (known) and p1 (unknown)
    # tangent vectors at endpoints are v0 (known) and v1 (known) (but magnitudes c0, c1 unknown)
    # control point is the intersection of the two tangent lines at the endpoints
    # arclength of curve = l (known)

    # solve for p1 = (x0, x1) (and thus the control point) given the constraints
    #              = (a * sin(theta), -a * cos(theta))           
    # define the objective
    # params are (a)


    theta = np.pi /20
    l = 1
    # get_quadratic_bezier_cp is redundant with the right angle one
    for bezier_fn in get_quadratic_bezier_right_angle, get_cubic_bezier:
        try:
            compute_objective = lambda params: objective(params, theta, l, bezier_fn)

            # initial guess
            x0 = [0.5, 0.5, 0.5]

            # max c1 is x0 sin theta / np.cos(theta)

            # define the constraints
            constraints = f'''
            x0 >= 0
            x0 <= {l}
            x1 >= 0.0
            x1 <= 10
            x2 >= 0.0
            x2 <= x0 * {np.sin(theta)} / {np.abs(np.cos(theta)) + 1e-6}
            '''

            generated_constraints = generate_constraint(generate_solvers(constraints, nvars=3))

            # solve the problem
            result = diffev2(compute_objective, x0=x0, constraints=generated_constraints, npop=40, gtol=10)
            plot_curve(result, theta, l, bezier_fn)
        except ValueError as e:
            print(e)
            continue
    plt.show(block=True)


if __name__ == "__main__":
    main()