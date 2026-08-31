import numpy as np
import scipy.sparse as sp
from .mpi_vector import KronVectorMPI
from pypardiso import PyPardisoSolver


def PCG(T, P, b, w0=None, kmax=100000, eps=1e-6, callback=None):
    """Preconditioned Conjuage Gradients with algebraic stopping criterium."""
    if w0 is None:
        if isinstance(b, KronVectorMPI):
            w = KronVectorMPI(b.dofs_distr)
        else:
            w = np.zeros(b.shape)
    else:
        w = w0

    iters = 0
    sq_rhs_norm = b.dot(b)
    if sq_rhs_norm == 0:
        return w, iters

    r = b - T @ w

    p = P @ r
    abs_r = r.dot(p)
    if abs_r < eps * eps:
        return w, iters
    for k in range(1, kmax):
        iters += 1
        t = T @ p
        alpha = abs_r / p.dot(t)
        w += alpha * p
        r -= alpha * t
        del t
        z = P @ r
        abs_r_old = abs_r
        abs_r = r.dot(z)
        if callback is not None:
            callback(w, abs_r, k)
        if abs_r < eps * eps:
            break
        beta = abs_r / abs_r_old
        p *= beta
        p += z
        del z
    return w, iters


# def build_preconditioner(A, M):
#     """
#     Build block-diagonal preconditioner:
#         diag(A^{-1}, M^{-1})
#     using sparse LU factorizations.
#     """

#     # Factorizations (you can swap for AMG if needed)
#     solver_A = PyPardisoSolver()
#     solver_M = PyPardisoSolver()
#     # A_lu = sp.splu(A.tocsc())
#     # M_lu = sp.splu(M.tocsc())

#     def apply_prec(x):
#         n_u = A.shape[0]
#         u = x[:n_u]
#         p = x[n_u:]

#         u_hat = solver_A.solve(A, u)
#         p_hat = solver_M.solve(M, p)

#         return np.concatenate([u_hat, p_hat])

#     n = A.shape[0] + M.shape[0]

#     return sp.LinearOperator((n, n), matvec=apply_prec)


def solve_saddle_minres(A, B, M, rhs, tol=1e-8, maxiter=5000):
    """
    Solve:
        [ A  B^T ] [u] = rhs
        [ B   0  ] [p]
    with MINRES and block-diagonal preconditioner.
    """

    n_u = A.shape[0]
    n_p = M.shape[0]

    zero = sp.csr_matrix((n_p, n_p))

    S = sp.bmat([[A, B], [B.T, zero]]).tocsr()

    print(S - S.transpose())

    S_op = sp.linalg.LinearOperator(S.shape, matvec=lambda x: S @ x, dtype=np.float64)

    solver_A = PyPardisoSolver()
    solver_M = PyPardisoSolver(mtype=2)
    # --- set matrix type (SPD) ---

    # --- configure iparm ---
    solver_A.iparm[0] = 1  # user-defined parameters enabled
    solver_A.iparm[1] = (
        3  # COLAMD (sometimes better than METIS for ill-conditioned problems)
    )
    solver_A.iparm[7] = 100
    solver_A.iparm[9] = 7  # perturbation (robustness)
    solver_A.iparm[10] = 1  # scaling enabled
    solver_A.iparm[12] = 1  # improved accuracy
    # solver_A.iparm[20] = 3
    solver_A.iparm[24] = 1  # scaling strategy
    solver_A.iparm[26] = 1  # check matrix symmetry
    # A_lu = sp.linalg.splu(A.tocsc())
    # M_lu = sp.linalg.splu(M.tocsc())
    # A = sp.triu(A, format="csr")
    M = sp.triu(M, format="csr")

    def apply_prec(x):
        n_u = A.shape[0]
        u = x[:n_u]
        p = x[n_u:]
        u = np.array(u, dtype=np.float64)
        p = np.array(p, dtype=np.float64)
        # u_hat = 0.5 * (A_lu.solve(u) + A_lu.solve(u, trans="T"))
        # p_hat = 0.5 * (M_lu.solve(p) + M_lu.solve(p, trans="T"))
        u_hat = solver_A.solve(A, u)
        p_hat = solver_M.solve(M, p)
        # u_hat = np.linalg.solve(A.toarray(), u)
        print("residual A:", np.linalg.norm(A @ u_hat - u))
        print("residual M:", np.linalg.norm(M @ p_hat - p))
        # return np.concatenate([u, p])
        return np.concatenate([u_hat, p_hat])

    prec = sp.linalg.LinearOperator((n_u + n_p, n_u + n_p), matvec=apply_prec)

    def callback(xk):
        r = rhs - S @ xk
        rel = np.linalg.norm(r) / np.linalg.norm(rhs)
        print(f"rel residual = {rel:.3e}")

    x, info = sp.linalg.minres(
        S, rhs, M=prec, rtol=tol, maxiter=maxiter, callback=callback, check=False
    )

    solver_A.free_memory()
    solver_M.free_memory()
    if info == 0:
        print("MINRES converged")
    else:
        print("MINRES stopped, info =", info)

    return x
