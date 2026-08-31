import numpy as np

from .mesh import construct_2d_square_mesh, construct_3d_cube_mesh, construct_interval


def interval(J_space, J_time=None):
    if J_time == None:
        J_time = J_space

    N_time = 2 ** int(J_time + 0.5)
    mesh_time = construct_interval(N=N_time)
    mesh_space = construct_interval(N=2 ** int(J_space + 0.5))
    bc = "start|end"
    from ngsolve import sin, cos, exp, x, y, CoefficientFunction, IfPos, sqrt, acos, pi

    data = {
        "u0": sin(np.pi * x),
        "q": lambda t: sin(np.pi * x) * cos(np.pi * t),
        "p": lambda t: -np.pi * sin(np.pi * x) * sin(np.pi * t),
        "v0": 0,
        "v": lambda t: -np.pi * sin(np.pi * x) * sin(np.pi * t),
        "v_t": -np.pi * sin(np.pi * x),
        "v_x": sin(np.pi * x),
        "sigma": lambda t: np.pi * cos(np.pi * x) * cos(np.pi * t),
        "sigma0": np.pi * cos(np.pi * x),
    }
    return mesh_space, bc, mesh_time, data, "interval"


def square(J_space, J_time=None):

    mesh_space, bc = construct_2d_square_mesh(nrefines=J_space)
    if J_time == None:
        J_time = J_space

    N_time = 2 ** int(J_time + 0.5)
    print("N_time:", N_time)
    mesh_time = construct_interval(N=N_time)

    import numpy as np
    from ngsolve import sin, cos, exp, x, y, CoefficientFunction, IfPos, sqrt, acos, pi

    # Solution u(t,x,y) = exp(−2 π^2 t) sin(πx) sin(πy) on [0,1]^2.
    # This is too difficult for the backward heat solver since uT is very small.
    factor = 1
    data = {
        "q0": sin(np.pi * x) * sin(np.pi * y),
        "q": lambda t: sin(np.pi * x) * sin(np.pi * y) * cos(np.sqrt(2) * np.pi * t),
        "p": lambda t: -np.sqrt(2)
        * np.pi
        * sin(np.pi * x)
        * sin(np.pi * y)
        * sin(np.sqrt(2) * np.pi * t),
        "v0": 0,
        "v": lambda t: -np.sqrt(2)
        * np.pi
        * sin(np.pi * x)
        * sin(np.pi * y)
        * sin(np.sqrt(2) * np.pi * t),
        "sigma": lambda t: CoefficientFunction(
            (
                np.pi * (cos(np.pi * x) * sin(np.pi * y)) * cos(np.sqrt(2) * np.pi * t),
                np.pi * (sin(np.pi * x) * cos(np.pi * y)) * cos(np.sqrt(2) * np.pi * t),
            )
        ),
        "sigma0": CoefficientFunction(
            (
                np.pi * (cos(np.pi * x) * sin(np.pi * y)),
                np.pi * (sin(np.pi * x) * cos(np.pi * y)),
            )
        ),
    }

    # # parameters
    # R = 0.25
    # r = sqrt((x-0.5)**2 + (y-0.5)**2)

    # # avoid division by zero
    # eps = 1e-12
    # r_safe = IfPos(r-eps, r, eps)

    # arg = lambda t:(r*r + (t)**2 - R*R) / (2*r_safe*t)

    # # clamp argument to [-1,1] to avoid NaNs
    # arg_clamped = lambda t:IfPos(arg(t)-1, 1,
    #                 IfPos(-1-arg(t), -1, arg(t)))

    # middle = lambda t: (1/pi) * acos(arg_clamped(t))

    # u_exact = lambda t: IfPos(R - (r + t), 1,
    #             IfPos((r - t) - R, 0, middle(t)))
    # u_0 = IfPos(R - r, 1, 0)
    # data = {
    #         'q0': u_0,
    #         'q': u_exact,
    #         'p': lambda t: 0
    # }
    # data = {
    #         'q0': factor * sin(np.pi * x) * sin(np.pi * y)
    #         }

    return mesh_space, bc, mesh_time, data, "square"


def cube(J_space, J_time=None):
    # Solution u(t,x,y) = exp(−3 π^2 t) sin(πx) sin(πy) \sin(πz) on [0,1]^3.
    mesh_space, bc = construct_3d_cube_mesh(nrefines=J_space)
    if not J_time:
        J_time = J_space

    N_time = 2 ** int(J_time + 0.5)
    mesh_time = construct_interval(N=N_time)

    from ngsolve import sin, x, y, z

    data = {"g": [], "u0": sin(np.pi * x) * sin(np.pi * y) * sin(np.pi * z)}
    return mesh_space, bc, mesh_time, data, "cube"


def problem_helper(problem, J_space, J_time=None):
    if problem == "square":
        return square(J_space, J_time)
    elif problem == "cube":
        return cube(J_space, J_time)
    elif problem == "interval":
        return interval(J_space, J_time)
    else:
        assert False
