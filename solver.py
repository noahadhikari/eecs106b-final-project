import mystic
from mystic.symbolic import generate_constraint, generate_solvers, generate_penalty, simplify
from mystic.solvers import diffev2
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt

def hermite_dist_objective(params, theta, l):
    a, b, c = params
    def x(t):
        return (-2*t**3 + 3*t**2) * a + (t**3 - t**2) * c
    def y(t):
        return (t**3 - 2*t**2 + t) * -1 + (-2*t**3 + 3*t**2) * b + (t**3 - t**2) * c * np.tan(theta)

    def dx(t):
        return (-6*t**2 + 6*t) * a + (3*t**2 - 2*t) * c
    
    def dy(t):
        return (3*t**2 - 4*t + 1) * -1 + (-6*t**2 + 6*t) * b + (3*t**2 - 2*t) * c * np.tan(theta)
    
    def length_func(t):
        return np.sqrt(dx(t)**2 + dy(t)**2)

    length, _ = quad(length_func, 0, 1)
    theta_f = np.arctan2(dy(1), dx(1))

    t = np.linspace(0, 1, 100)
    cog_y = np.mean(y(t))

    print(length, theta_f)
    

    cog_weight = 0.01
    return (length - l)**2 + (np.sin(theta) - np.sin(theta_f))**2 + (np.cos(theta) - np.cos(theta_f))**2 + cog_weight * cog_y
    
def plot_hermite(a, b, c, theta, l):
    def x(t):
        return (-2*t**3 + 3*t**2) * a + (t**3 - t**2) * c
    def y(t):
        return (t**3 - 2*t**2 + t) * -1 + (-2*t**3 + 3*t**2) * b + (t**3 - t**2) * c * np.tan(theta)
    
    t = np.linspace(0, 1, 100)
    plt.plot(x(t), y(t))
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
    # Define p0 = (0, 0), m0 = (0, -1), p1 = (a, b), m1 = c(x'(1), y'(1))
    #
    # Require angle at the endpoint to be theta
    # need to solve for a, b, c
    
    # given variables
    theta = 0
    l = 1

    # define the objective
    objective = lambda params: hermite_dist_objective(params, theta, l)

    constraints = f'''
    x0 >= 0
    x0 <= {l}
    x1 >= -{l}
    x1 <= {l}
    x2 >= 0
    '''

    generated_constraints = generate_constraint(generate_solvers(constraints))

    # initial guess
    x0 = [0.5, 0.5, 0.5]

    # solve
    result = diffev2(objective, x0=x0, bounds=[(-l, l), (-l, l), (-l, l)], constraints=generated_constraints, npop=40, gtol=10)

    print(result)
    plot_hermite(*result, theta, l)

    
if __name__ == "__main__":
    main()