import argparse

import numpy as np
import scipy.sparse as sp
from ngsolve import (H1, L2, InnerProduct, Preconditioner, ds, dx, grad,
                     ngsglobals, preconditioners)

from source.linalg import PCG
from source.linop import (AsLinearOperator, BlockDiagLinOp, CompositeLinOp,
                          KronLinOp)
from source.ngsolve_helper import BilForm, KronBF, KronFES, KronLF
from source.problem import problem_helper
from source.wavelets import WaveletTransformOp

ngsglobals.msg_level = 0


class BackwardHeatEquation:

    def __init__(self,
                 J_space=2,
                 J_time=None,
                 problem='square',
                 precond='multigrid',
                 epsilon=0,
                 d_orderY=0,
                 alpha=0.3,
                 order=1):
        mesh_space, bc, mesh_time, data, fn = problem_helper(problem,
                                                             J_space=J_space,
                                                             J_time=J_time)
        self.mesh_space = mesh_space
        d = self.mesh_space.dim
        self.data = data
        # Building fespaces X^\delta and Y^\delta.
        self.X = KronFES(H1(mesh_time, order=order), 
                         H1(mesh_space, order=order, dirichlet=bc))
        self.Y = KronFES(L2(mesh_time, order=order),
                         H1(mesh_space, order=order + d_orderY, dirichlet=bc))
        self.N = len(self.X.time.fd)
        self.M = len(self.X.space.fd)

        # Building the ngsolve spactime-bilforms.
        dt=dx        
        A_bf = KronBF(self.Y, self.Y, lambda u, v: u * v * dt,
                      lambda u, v: InnerProduct(grad(u), grad(v)) * dx)
        A_X_bf = KronBF(self.X, self.X, lambda u, v: u * v * dt,
                        lambda u, v: InnerProduct(grad(u), grad(v)) * dx)
        B1_bf = KronBF(self.X, self.Y, lambda u, v: grad(u) * v * dt,
                       lambda u, v: u * v * dx)
        B2_bf = KronBF(self.X, self.Y, lambda u, v: u * v * dt,
                       lambda u, v: InnerProduct(grad(u), grad(v)) * dx)
        G_end_bf = KronBF(self.X, self.X, lambda u, v: u * v * ds('end'),
                          lambda u, v: u * v * dx)
        G_start_bf = KronBF(self.X, self.X, lambda u, v: epsilon * u * v * ds('start'),
                           lambda u, v: u * v * dx)

        self.B = B1_bf.assemble() + B2_bf.assemble()
        self.BT = B1_bf.transpose() + B2_bf.transpose()
        self.G_end = G_end_bf.assemble()
        self.G_start = G_start_bf.assemble()

        # Preconditioner on Y.
        Kinv_time_pc = Preconditioner(A_bf.time.bf, 'direct')
        Kinv_space_pc = Preconditioner(A_bf.space.bf, precond)
        A_bf.assemble()
        Kinv_time = AsLinearOperator(Kinv_time_pc.mat, self.Y.time.fd)
        Kinv_space = AsLinearOperator(Kinv_space_pc.mat, self.Y.space.fd)
        self.K = KronLinOp(Kinv_time, Kinv_space)

        # --- Wavelet transform ---
        W_t = WaveletTransformOp(J_time)
        self.W = KronLinOp(W_t, sp.eye(len(self.X.space.fd)))
        self.WT = KronLinOp(W_t.T, sp.eye(len(self.X.space.fd)))

        # Preconditioner on X.
        A_X_bf.assemble()
        self.C_j = []
        self.alpha = alpha
        for j in range(J_time + 1):
            bf = BilForm(self.X.space,
                         bilform_lambda=lambda u, v:
                         (2**j * u * v + self.alpha * grad(u) * grad(v)) * dx)
            C = Preconditioner(bf.bf, precond)
            bf.bf.Assemble()
            self.C_j.append(AsLinearOperator(C.mat, self.X.space.fd))

        self.CAC_j = [
            CompositeLinOp([self.C_j[j], A_X_bf.space.mat, self.C_j[j]])
            for j in range(J_time + 1)
        ]
        self.P = BlockDiagLinOp([self.CAC_j[j] for j in W_t.levels])


        # Schur-complement operator.
        self.S = sp.linalg.LinearOperator(
            self.G_end.shape,
            matvec=lambda v: self.BT @ self.K @ self.B @ v + self.G_start @ v + self.G_end @ v)
        self.WT_S_W = self.WT @ self.S @ self.W

        # Calculate rhs.
        self.g_vec = np.zeros(self.K.shape[0])
        for g in data['g']:
            g_lf = KronLF(self.Y, lambda v: g[0] * v * dt, lambda v: g[1] * v * dx)
            g_lf.assemble()
            self.g_vec += g_lf.vec
        uT_lf = KronLF(self.X, lambda v: v * ds('end'),
                       lambda v: data['uT'] * v * dx)
        uT_lf.assemble()

        self.f = self.BT @ self.K @ self.g_vec + uT_lf.vec


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Solve heatequation using ngsolve.')
    parser.add_argument('--problem',
                        default='square',
                        help='problem type (square, ns)')
    parser.add_argument('--J_time',
                        type=int,
                        default=5,
                        help='number of time refines')
    parser.add_argument('--J_space',
                        type=int,
                        default=6,
                        help='number of space refines')
    parser.add_argument(
        '--precond',
        default="multigrid",
        help='type of ngsolve preconditioner, e.g. direct or multigrid.')
    parser.add_argument('--alpha',
                        type=float,
                        default=0.3,
                        help='Alpha value used in the preconditioner for X.')
    parser.add_argument('--epsilon',
                        type=float,
                        default=0,
                        help='Epsilon value used in for regularisation.')
    parser.add_argument('--d_orderY',
                        type=int,
                        default=0,
                        help='Extra order of the finite element space Y. ' \
                        'Should be at most d + 1, since we can prove stability in this case')
    args = parser.parse_args()
    order = 1  # Higher order requires a different wavelet transform.

    print('Arguments: {}'.format(args))
    print(
        '\n\nCreating HeatEquation with {} time refines and {} space refines.'.
        format(args.J_time, args.J_space))
    heat_eq = BackwardHeatEquation(J_time=args.J_time,
                           J_space=args.J_space,
                           problem=args.problem,
                           precond=args.precond,
                           epsilon=args.epsilon,
                           d_orderY=args.d_orderY,
                           alpha=args.alpha)
    print('Size of time mesh: {} dofs. Size of space mesh: {} dofs'.format(
        heat_eq.N, heat_eq.M))

    def cb(w, residual, k):
        print(residual, end='\n', flush=True)

    print("Solving: ", end='')
    w, iters = PCG(heat_eq.WT_S_W,
                   heat_eq.P,
                   heat_eq.WT @ heat_eq.f, eps=1e-6,
                   callback=cb)
    u = heat_eq.W @ w
    res = heat_eq.f - heat_eq.S @ u
    error_alg = res @ (heat_eq.P @ res)

    gminBu = heat_eq.g_vec - heat_eq.B @ u
    error_Yprime = gminBu @ (heat_eq.K @ gminBu)
    print("Done in {}  PCG steps. "
          "X-norm algebraic error: {}. "
          "Error in Yprime: {}\n".format(iters, error_alg, error_Yprime))

    from ngsolve import GridFunction
    from ngsolve import Integrate
    from ngsolve import sin, cos, x, y, CoefficientFunction

    u_t = lambda t: heat_eq.X.solution_at_t(t, u)
    exact_u = heat_eq.data['u']
    grad_exact = heat_eq.data['grad_u']

    t = 0
    ut = u_t(t)
    err = Integrate((ut - exact_u(t))**2, mesh = heat_eq.mesh_space)
    err = np.sqrt(err)
    print("L2 error at t={}:".format(t), "{}".format(err))

    t = 0.1
    ut = u_t(t)
    err = Integrate((ut - exact_u(t))**2, mesh = heat_eq.mesh_space)
    err = np.sqrt(err)
    print("L2 error at t={}:".format(t), "{}".format(err))

    error_space = lambda t:Integrate(InnerProduct(u_t(t) - exact_u(t), u_t(t) - exact_u(t)), mesh=heat_eq.mesh_space)

    print("L2(J x Omega)-error = {}".format(
        np.sqrt(heat_eq.X.Integrate_time(error_space))))

    error_grad = lambda t:Integrate(InnerProduct(grad(u_t(t)) - grad_exact(t), grad(u_t(t)) - grad_exact(t)), mesh=heat_eq.mesh_space)
    print("L2(J; H1(Omega))-error = {}".format(
        np.sqrt(heat_eq.X.Integrate_time(error_grad))))
