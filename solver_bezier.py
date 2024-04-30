import numpy as np
from bezier.curve import Curve

from mystic.symbolic import generate_constraint, generate_solvers
from mystic.solvers import diffev2

import matplotlib.pyplot as plt


def get_cubic_bezier(params, theta):
    x, y, c0, c1 = params

    p0 = np.array([0, 0])
    v0 = c0 * np.array([0, -1])
    v1 = c1 * np.array([np.cos(theta), np.sin(theta)])
    p1 = np.array([x, y])

    # set the control points to be 1 unit of time in velocity vector from the endpoint

    pc0 = p0 + v0
    pc1 = p1 - v1

    # define the curve
    nodes = np.asfortranarray([p0, pc0, pc1, p1]).T
    curve = Curve(nodes, degree=3)

    return curve

def get_quadratic_bezier(params, theta):
    x, y, c0, c1 = params

    p0 = np.array([0, 0])
    v0 = np.array([0, -1])
    v1 = np.array([np.cos(theta), np.sin(theta)])
    p1 = np.array([x, y])

    # find the intersection between the two tangent lines
    # f0(t) = p0 + tv0
    # f1(s) = p1 + sv1
    # f0(t) = f1(s) => p0 + tv0 = p1 + sv1
    # => p0 - p1 = sv1 - tv0
    # => p0 - p1 = [v1, -v0] [s, t]^T
    # => [v1, -v0]^-1 [p0 - p1] = [s, t]^T

    A = np.array([v1, -v0]).T
    if np.linalg.det(A) < 1e-6:
        raise ValueError("Lines are parallel, cannot find intersection point.")
    s, t = np.linalg.inv(A) @ (p0 - p1)

    pc = p0 + t * v0

    # define the curve
    nodes = np.asfortranarray([p0, pc, p1]).T
    curve = Curve(nodes, degree=2)

    return curve

def objective(params, theta, l):
    def obj(curve):
        p1 = curve.evaluate(1.)
        theta_f = np.arctan2(p1[0], -p1[1])

        return 10 * (curve.length - l) ** 2 + (np.sin(theta) - np.sin(theta_f)) ** 2 + (np.cos(theta) - np.cos(theta_f)) ** 2

    cubic = get_cubic_bezier(params, theta)

    return obj(cubic)

def plot_curve(params, theta):
    curve = get_cubic_bezier(params, theta)
    curve.plot(num_pts=100)
    plt.show(block=True)

def main():

    theta = np.pi / 2
    l = 1

    p0 = np.array([0, 0])
    v0 = np.array([0, -1])

    v1 = np.array([np.cos(theta), np.sin(theta)])

    # f : R -> R^2, t in [0, 1] -> (x(t), y(t)) given by a quadratic bezier curve
    # endpoints are p0 (known) and p1 (unknown)
    # tangent vectors at endpoints are v0 (known) and v1 (known) (but magnitudes c0, c1 unknown)
    # control point is the intersection of the two tangent lines at the endpoints
    # arclength of curve = l (known)

    # solve for p1 = (x0, x1) (and thus the control point) given the constraints

    # define the objective
    # params are (x0, x1, c0, c1)
    compute_objective = lambda params: objective(params, theta, l)

    # initial guess
    x0 = [0.5, -0.5, 1, 1]

    # max c1 is x0 / np.cos(theta)

    # define the constraints
    constraints = f'''
    x0 >= 0
    x0 <= {l}
    x1 >= -{l}
    x1 <= 0
    x2 >= 0.0
    x2 <= 10
    x3 >= 0.0
    x3 <= x0 / {np.abs(np.cos(theta)) + 1e-6}
    '''

    generated_constraints = generate_constraint(generate_solvers(constraints, nvars=4))

    # solve the problem
    result = diffev2(compute_objective, x0=x0, constraints=generated_constraints, npop=40, gtol=100)
    plot_curve(result, theta)



if __name__ == "__main__":
    main()