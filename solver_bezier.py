import numpy as np
from bezier.curve import Curve

from mystic.symbolic import generate_constraint, generate_solvers
from mystic.solvers import diffev2

import matplotlib.pyplot as plt


def get_curve(params, theta):
    x, y = params

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
    s, t = np.linalg.inv(A) @ (p0 - p1)

    pc = p0 + t * v0

    # define the curve
    nodes = np.asfortranarray([p0, pc, p1]).T
    curve = Curve(nodes, degree=2)

    return curve

def objective(params, theta, l):
    curve = get_curve(params, theta)
    return (curve.length - l) ** 2

def plot_curve(params, theta):
    curve = get_curve(params, theta)
    curve.plot(100)
    plt.show(block=True)

def main():

    theta = np.pi / 4
    l = 3

    p0 = np.array([0, 0])
    v0 = np.array([0, -1])

    v1 = np.array([np.cos(theta), np.sin(theta)])

    # f : R -> R^2, t in [0, 1] -> (x(t), y(t)) given by a quadratic bezier curve
    # endpoints are p0 (known) and p1 (unknown)
    # tangent vectors at endpoints are v0 (known) and v1 (known) (but magnitudes unknown)
    # control point is the intersection of the two tangent lines at the endpoints
    # arclength of curve = l (known)

    # solve for p1 = (x0, x1) (and thus the control point) given the constraints

    # define the objective
    compute_objective = lambda params: objective(params, theta, l)

    # initial guess
    x0 = [0.5, -0.5]

    # define the constraints
    constraints = f'''
    x0 >= 0
    x0 <= {l}
    x1 >= -{l}
    x1 <= {0}
    '''

    generated_constraints = generate_constraint(generate_solvers(constraints))

    # solve the problem
    result = diffev2(compute_objective, x0=x0, constraints=generated_constraints, npop=40, gtol=10)
    plot_curve(result, theta)



if __name__ == "__main__":
    main()