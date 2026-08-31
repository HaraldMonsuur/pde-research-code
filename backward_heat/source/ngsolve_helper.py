import numpy as np
import scipy.sparse as sp
from ngsolve import BilinearForm, LinearForm, GridFunction

from .linop import KronLinOp


class KronFES:
    """Wrapper around ngsolve::FESpace."""

    def __init__(self, fes_time, fes_space):
        self.time = fes_time
        self.space = fes_space
        self.time.fd = [i for (i, free) in enumerate(fes_time.FreeDofs()) if free]
        self.space.fd = [i for (i, free) in enumerate(fes_space.FreeDofs()) if free]

    def Coeff_space(self):
        I = [i for (i, free) in enumerate(self.space.FreeDofs()) if free]
        J = np.array(range(0, len(I)))
        data = np.ones(len(I))
        mat = sp.csr_matrix((data, (I, J)), shape=(self.space.ndof, len(J)))
        return mat

    def Coeff_time(self):
        I = [i for (i, free) in enumerate(self.time.FreeDofs()) if free]
        J = np.array(range(0, len(I)))
        data = np.ones(len(I))
        mat = sp.csr_matrix((data, (I, J)), shape=(self.time.ndof, len(I)))
        return mat

    def solution_at_t(self, t, u):
        Coeff_space = self.Coeff_space()
        nintervals = self.time.ndof - 1
        h_t = 1.0 / nintervals
        i = int(t / h_t)
        if i == nintervals:
            i = nintervals - 1
            c0 = 0
            c1 = 1.0
        else:
            c1 = (t - i * h_t) / h_t
            c0 = 1.0 - c1
        ut = GridFunction(self.space)
        ut.vec.data = Coeff_space @ (
            u[Coeff_space.shape[1] * i : Coeff_space.shape[1] * (i + 1)] * c0
            + u[Coeff_space.shape[1] * (i + 1) : Coeff_space.shape[1] * (i + 2)] * c1
        )
        return ut

    def solution_at_t_vec(self, t, u):
        ut = self.solution_at_t(t, u)
        return ut.vec.data.FV().NumPy()[:][self.space.FreeDofs()]

    def Integrate_time(self, u_t=None):
        """Integrate u_t over the time mesh."""
        # This function assumes a uniform mesh!
        integral = 0.0
        # quad_points = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
        # quad_weights = np.array([1.0, 1.0])
        quad_points = np.array([-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)])
        quad_weights = np.array([5 / 9, 8 / 9, 5 / 9])
        nintervals = self.time.ndof - 1
        print("nintervals:", nintervals)
        for i in range(nintervals):
            x0 = i / nintervals
            x1 = (i + 1) / nintervals
            h = (x1 - x0) / 2
            xm = (x1 + x0) / 2
            for xi, wi in zip(quad_points, quad_weights):
                x_eval = xm + h * xi
                if u_t is not None:
                    u_eval = u_t(x_eval)
                else:
                    u_eval = 0
                integral += wi * h * u_eval

        return integral


class BilForm:
    """Wrapper class around ngsolve for easier creation of bilforms."""

    def __init__(self, fes_in, fes_out=None, bilform_lambda=None):
        self.fes_in = fes_in
        self.fes_out = fes_out if fes_out else fes_in
        if self.fes_in is self.fes_out:
            self.bf = BilinearForm(self.fes_in, symmetric=False, check_unused=False)
        else:
            self.bf = BilinearForm(
                self.fes_in, self.fes_out, symmetric=False, check_unused=False
            )
        if bilform_lambda == None:
            return
        self.bf += bilform_lambda(
            self.fes_in.TrialFunction(), self.fes_out.TestFunction()
        )

    def assemble(self):
        self.bf.Assemble()
        mat = sp.csr_matrix(self.bf.mat.CSR())
        self.mat = (
            mat[self.fes_out.FreeDofs(), :].tocsc()[:, self.fes_in.FreeDofs()].tocsr()
        )
        self.mat.eliminate_zeros()
        return self.mat


class KronBF:
    """Helper class that represents a kronecker ngsolve bilform."""

    def __init__(
        self, fes_in, fes_out=None, bilform_time_lambda=None, bilform_space_lambda=None
    ):
        self.fes_in = fes_in
        self.fes_out = fes_out if fes_out else fes_in
        self.time = BilForm(self.fes_in.time, self.fes_out.time, bilform_time_lambda)
        self.space = BilForm(
            self.fes_in.space, self.fes_out.space, bilform_space_lambda
        )

    def assemble(self):
        self.time.assemble()
        self.space.assemble()
        return KronLinOp(self.time.mat, self.space.mat)

    def as_operator(self):
        return KronLinOp(self.time.mat, self.space.mat)

    def sp_mat(self):
        return sp.kron(self.time.mat, self.space.mat)

    def transpose(self):
        return KronLinOp(self.time.mat.T, self.space.mat.T)


class LinForm:
    """Wrapper around ngsolve::LinearForm."""

    def __init__(self, fes, linform_lambda=None):
        self.fes = fes
        self.lf = LinearForm(fes)
        if linform_lambda == None:
            return
        self.lf += linform_lambda(self.fes.TestFunction())

    def assemble(self):
        self.lf.Assemble()
        self.vec = self.lf.vec.FV().NumPy()[self.fes.fd]
        return self.vec


class KronLF:
    """Kronecker product of two LinForms."""

    def __init__(self, fes, time_lambda=None, space_lambda=None):
        self.time = LinForm(fes.time, time_lambda)
        self.space = LinForm(fes.space, space_lambda)

    def assemble(self):
        self.time.assemble()
        self.space.assemble()
        self.vec = np.kron(self.time.vec, self.space.vec)
        return self.vec
