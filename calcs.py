from sympy.geometry import Curve
from sympy.solvers import solve
from sympy.vector import CoordSys3D
from sympy import symbols, Function, diff, integrate, tan, pi, sqrt
import numpy as np



def main():

    # f : R -> R^2, t in [0, 1] -> (x(t), y(t))
    # constraints:
    # f(0) = 0
    # x'(0) = 0
    # y'(0) = -1
    # y'(1) = x'(1) tan theta
    # arclength of curve = l for predefined l. define l(t) = int_0^t sqrt(x'(s)^2 + y'(s)^2) ds
    #       so l(1) = l
    # use hermite interpolation

    # define symbols
    t = symbols('t')
    x = Function('x')(t)
    y = Function('y')(t)
    f = Curve([x, y], (t, 0, 1))
    l = Function('l')(t)
    dx = diff(x, t)
    dy = diff(y, t)

    theta_f = pi/4
    l_f = 1

    # define the curve
    l = integrate(sqrt(dx**2 + dy**2), (t, 0, t))

    # define the constraints
    constraints = [
        x.subs(t, 0),
        dx.subs(t, 0),
        dy.subs(t, 0) + 1,
        dy.subs(t, 1) - dx.subs(t, 1) * tan(theta_f),
        l.subs(t, 1) - l_f
    ]

    # solve the constraints??
    solutions = solve(constraints)



if __name__ == '__main__':
    main()