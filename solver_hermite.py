import mystic
from mystic.symbolic import generate_constraint, generate_solvers, generate_penalty, simplify
from mystic.solvers import diffev2
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt


def derivative_hermite(p0, m0, p1, m1):
    def dp(t):
        return (6*t**2 - 6*t) * p0 + (3*t**2 - 4*t + 1) * m0 + (-6*t**2 + 6*t) * p1 + (3*t**2 - 2*t) * m1
    return dp

# returns the hermite interpolation function on [0, 1]
def hermite(p0, m0, p1, m1):
    def p(t):
        return (2*t**3 - 3*t**2 + 1) * p0 + (t**3 - 2*t**2 + t) * m0 + (-2*t**3 + 3*t**2) * p1 + (t**3 - t**2) * m1
    return p

def length(df, a, b):
    length_func = lambda t: np.linalg.norm(df(t))
    return quad(length_func, a, b)[0]


def get_hermite_coeffs(params, theta):
    a, b, c = params
    p0 = np.array([0, 0])
    m0 = np.array([0, -a])
    p1 = np.array([b * np.sin(theta), -b * np.cos(theta)])
    m1 = np.array([c * np.cos(theta), c * np.sin(theta)])

    return p0, m0, p1, m1


def objective(params, theta, l):
    p0, m0, p1, m1 = get_hermite_coeffs(params, theta)
    f = hermite(p0, m0, p1, m1)
    df = derivative_hermite(p0, m0, p1, m1)

    return (length(df, 0, 1) - l) ** 2

def plot_hermite(params, theta, l_max):

    ax = plt.subplot(111)
    ax.set_aspect('equal', 'box')
    plt.xlim(-0.05, l_max)
    plt.ylim(-l_max, 0.5)
    # draw vertical and angle lines
    vert_line = np.array([[0, 0], [0, -l_max]])
    angle_line = l_max * np.array([[0, 0], [np.sin(theta), -np.cos(theta)]])
    
    plt.plot(vert_line[:, 0], vert_line[:, 1], 'k-')
    plt.plot(angle_line[:, 0], angle_line[:, 1], 'k-')

    t = np.linspace(0, 1, 100)

    p0, m0, p1, m1 = get_hermite_coeffs(params, theta)
    f = hermite(p0, m0, p1, m1)
    ft = np.array([f(ti) for ti in t])
    plt.plot(ft[:, 0], ft[:, 1], 'r-')
    plt.title('Hermite Curve Model of Soft Grippers')
    plt.xlabel('Position in x')
    plt.ylabel('Position in y')    
    plt.show(block=True)

def main():
    # f : R -> R^2, t in [0, 1] -> (x(t), y(t))
    # constraints:
    # f(0) = 0
    # x'(0) = 0
    # y'(0) = -1
    # x'(1) = cos theta (given theta)
    # y'(1) = sin theta (given theta)
    # arclength of curve = l (given l). define l(t) = int_0^t sqrt(x'(s)^2 + y'(s)^2) ds
    #       so l(1) = l
    # use hermite interpolation, so f(t) = (2t^3 - 3t^2 + 1) * p0 + (t^3 - 2t^2 + t) * m0 + (-2t^3 + 3t^2) * p1 + (t^3 - t^2) * m1
    # Define p0 = (0, 0), m0 = (0, -a), p1 = (b sin theta, -b cos theta), m1 = (c cos theta, c sin theta)
    #
    # require length of the curve to be l
    # need to solve for a, b, c
    
    # given variables
    theta = 20*(np.pi / 180)
    l = 1

    # define the objective
    compute_objective = lambda params: objective(params, theta, l)

    constraints = f'''
    x0 >= 0
    x0 <= {l * 100}
    x1 >= 0
    x1 <= {l}
    x2 >= 0
    x2 <= {l * 100}
    '''

    generated_constraints = generate_constraint(generate_solvers(constraints, nvars=3))

    # initial guess
    x0 = [0.5, 0.5, 0.5]

    # solve
    params = diffev2(compute_objective, x0=x0, constraints=generated_constraints, npop=40, gtol=10)

    plot_hermite(params, theta, l)

    
if __name__ == "__main__":
    main()