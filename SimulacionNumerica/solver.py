"""
Solucionador Navier-Stokes incompresible 2D en formulacion funcion de
corriente - vorticidad (psi-omega), resuelto con Newton-Raphson y
Jacobiano analitico disperso (CSR).

Trabajamos en unidades de malla (h=1, espaciado entre nodos indexado).
El numero de Reynolds R es el parametro de control del termino convectivo.

Ecuaciones (nodo interior de fluido):
  Poisson (lineal):      psi_{i+1,j}+psi_{i-1,j}+psi_{i,j+1}+psi_{i,j-1}
                          - 4 psi_{i,j} + omega_{i,j} = 0
  Transporte (no lineal): om_{i+1,j}+om_{i-1,j}+om_{i,j+1}+om_{i,j-1}
                          - 4 om_{i,j}
                          - (R/4)[ (psi_{i,j+1}-psi_{i,j-1})(om_{i+1,j}-om_{i-1,j})
                                 - (psi_{i+1,j}-psi_{i-1,j})(om_{i,j+1}-om_{i,j-1}) ] = 0
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class Grid:
    def __init__(self, Nx, Ny, V0=1.0, h=1.0,
                 obs1=(0, 6, 12, 16),      # (i0,i1,j0,j1) inclusive  -> top-left
                 obs2=(76, 84, 0, 8)):     # bottom-center
        """Nx, Ny = numero de NODOS; h = espaciado fisico entre nodos."""
        self.Nx, self.Ny, self.V0, self.h = Nx, Ny, V0, h
        self.obs1 = obs1
        self.obs2 = obs2
        self._classify()
        self._index()

    # ---- geometria de obstaculos -------------------------------------
    def in_obs(self, i, j):
        for (i0, i1, j0, j1), _psi in [(self.obs1, self.V0*(self.Ny-1)*self.h),
                                       (self.obs2, 0.0)]:
            if i0 <= i <= i1 and j0 <= j <= j1:
                return True
        return False

    def obs_psi(self, i, j):
        i0, i1, j0, j1 = self.obs1
        if i0 <= i <= i1 and j0 <= j <= j1:
            return self.V0*(self.Ny-1)       # pegado a la tapa superior
        return 0.0                            # obstaculo 2 -> pegado al fondo

    # ---- clasificacion de nodos --------------------------------------
    # tipos: 'inlet','outlet','wall','surf','osolid','fluid'
    def _classify(self):
        Nx, Ny = self.Nx, self.Ny
        self.type = np.empty((Nx, Ny), dtype=object)
        self.solid = np.zeros((Nx, Ny), dtype=bool)
        for i in range(Nx):
            for j in range(Ny):
                if self.in_obs(i, j):
                    self.solid[i, j] = True
        for i in range(Nx):
            for j in range(Ny):
                if self.solid[i, j]:
                    # superficie si tiene algun vecino de fluido (no solido)
                    surf = False
                    for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
                        ii, jj = i+di, j+dj
                        if 0 <= ii < Nx and 0 <= jj < Ny and not self.solid[ii, jj]:
                            surf = True
                    self.type[i, j] = 'surf' if surf else 'osolid'
                elif i == 0:
                    self.type[i, j] = 'inlet'
                elif j == 0 or j == Ny-1:
                    self.type[i, j] = 'wall'
                elif i == Nx-1:
                    self.type[i, j] = 'outlet'
                else:
                    self.type[i, j] = 'fluid'

    # ---- valores fijos y mapas de incognitas -------------------------
    def _index(self):
        Nx, Ny = self.Nx, self.Ny
        self.psi_fix = np.zeros((Nx, Ny))
        self.om_fix = np.zeros((Nx, Ny))
        self.psi_known = np.zeros((Nx, Ny), dtype=bool)
        self.om_known = np.zeros((Nx, Ny), dtype=bool)

        for i in range(Nx):
            for j in range(Ny):
                t = self.type[i, j]
                if t == 'inlet':
                    self.psi_fix[i, j] = self.V0*j*self.h; self.psi_known[i, j] = True
                    self.om_fix[i, j] = 0.0;        self.om_known[i, j] = True
                elif t == 'wall':
                    self.psi_fix[i, j] = 0.0 if j == 0 else self.V0*(Ny-1)*self.h
                    self.psi_known[i, j] = True     # omega incognita (Thom)
                elif t == 'surf':
                    self.psi_fix[i, j] = self.obs_psi(i, j)
                    self.psi_known[i, j] = True     # omega incognita (Thom)
                elif t == 'osolid':
                    self.psi_fix[i, j] = self.obs_psi(i, j); self.psi_known[i, j] = True
                    self.om_fix[i, j] = 0.0;                 self.om_known[i, j] = True
                # outlet y fluid: ambas incognitas

        # indices globales: primero psi, luego omega
        self.ipsi = -np.ones((Nx, Ny), dtype=int)
        self.iom = -np.ones((Nx, Ny), dtype=int)
        c = 0
        for i in range(Nx):
            for j in range(Ny):
                if not self.psi_known[i, j]:
                    self.ipsi[i, j] = c; c += 1
        for i in range(Nx):
            for j in range(Ny):
                if not self.om_known[i, j]:
                    self.iom[i, j] = c; c += 1
        self.N = c

    # ---- fluid neighbors para Thom -----------------------------------
    def fluid_neighbors(self, i, j):
        out = []
        for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
            ii, jj = i+di, j+dj
            if 0 <= ii < self.Nx and 0 <= jj < self.Ny and not self.solid[ii, jj]:
                out.append((ii, jj))
        return out


def reconstruct(g, x):
    Psi = g.psi_fix.copy()
    Om = g.om_fix.copy()
    Psi[~g.psi_known] = x[g.ipsi[~g.psi_known]]
    Om[~g.om_known] = x[g.iom[~g.om_known]]
    return Psi, Om


def residual(g, x, R):
    Psi, Om = reconstruct(g, x)
    F = np.zeros(g.N)
    Nx, Ny = g.Nx, g.Ny
    for i in range(Nx):
        for j in range(Ny):
            t = g.type[i, j]
            # --- ecuacion de psi ---
            ip = g.ipsi[i, j]
            if ip >= 0:
                if t == 'outlet':
                    F[ip] = Psi[i, j] - Psi[i-1, j]
                else:  # fluid
                    F[ip] = (Psi[i+1, j]+Psi[i-1, j]+Psi[i, j+1]+Psi[i, j-1]
                             - 4*Psi[i, j] + g.h*g.h*Om[i, j])
            # --- ecuacion de omega ---
            io = g.iom[i, j]
            if io >= 0:
                if t == 'outlet':
                    F[io] = Om[i, j] - Om[i-1, j]
                elif t in ('wall', 'surf'):
                    nb = g.fluid_neighbors(i, j)
                    s = sum(Psi[ii, jj] - Psi[i, j] for ii, jj in nb)
                    F[io] = Om[i, j] + 2.0*s/(len(nb)*g.h*g.h)
                else:  # fluid
                    conv = ((Psi[i, j+1]-Psi[i, j-1])*(Om[i+1, j]-Om[i-1, j])
                            - (Psi[i+1, j]-Psi[i-1, j])*(Om[i, j+1]-Om[i, j-1]))
                    F[io] = (Om[i+1, j]+Om[i-1, j]+Om[i, j+1]+Om[i, j-1]
                             - 4*Om[i, j] - 0.25*R*conv)
    return F


def jacobian(g, x, R):
    Psi, Om = reconstruct(g, x)
    Nx, Ny = g.Nx, g.Ny
    rows, cols, vals = [], [], []

    def add(r, i, j, which, coef):
        idx = g.ipsi[i, j] if which == 'p' else g.iom[i, j]
        if idx >= 0:
            rows.append(r); cols.append(idx); vals.append(coef)

    for i in range(Nx):
        for j in range(Ny):
            t = g.type[i, j]
            ip = g.ipsi[i, j]
            if ip >= 0:
                if t == 'outlet':
                    add(ip, i, j, 'p', 1.0)
                    add(ip, i-1, j, 'p', -1.0)
                else:
                    add(ip, i, j, 'p', -4.0)
                    add(ip, i+1, j, 'p', 1.0); add(ip, i-1, j, 'p', 1.0)
                    add(ip, i, j+1, 'p', 1.0); add(ip, i, j-1, 'p', 1.0)
                    add(ip, i, j, 'o', g.h*g.h)
            io = g.iom[i, j]
            if io >= 0:
                if t == 'outlet':
                    add(io, i, j, 'o', 1.0)
                    add(io, i-1, j, 'o', -1.0)
                elif t in ('wall', 'surf'):
                    nb = g.fluid_neighbors(i, j)
                    add(io, i, j, 'o', 1.0)
                    for ii, jj in nb:
                        add(io, ii, jj, 'p', 2.0/(len(nb)*g.h*g.h))
                else:  # fluid
                    dpsy = Psi[i, j+1]-Psi[i, j-1]
                    dpsx = Psi[i+1, j]-Psi[i-1, j]
                    domx = Om[i+1, j]-Om[i-1, j]
                    domy = Om[i, j+1]-Om[i, j-1]
                    # d/d omega
                    add(io, i, j, 'o', -4.0)
                    add(io, i+1, j, 'o', 1.0 - 0.25*R*dpsy)
                    add(io, i-1, j, 'o', 1.0 + 0.25*R*dpsy)
                    add(io, i, j+1, 'o', 1.0 + 0.25*R*dpsx)
                    add(io, i, j-1, 'o', 1.0 - 0.25*R*dpsx)
                    # d/d psi
                    add(io, i, j+1, 'p', -0.25*R*domx)
                    add(io, i, j-1, 'p', +0.25*R*domx)
                    add(io, i+1, j, 'p', +0.25*R*domy)
                    add(io, i-1, j, 'p', -0.25*R*domy)
    J = sp.csr_matrix((vals, (rows, cols)), shape=(g.N, g.N))
    return J


def newton(g, R, x0=None, tol=1e-8, maxit=60, verbose=False, backtrack=True):
    x = np.zeros(g.N) if x0 is None else x0.copy()
    hist = []
    for k in range(maxit):
        F = residual(g, x, R)
        nrm = np.linalg.norm(F, np.inf)
        hist.append(nrm)
        if verbose:
            print(f"    it {k:2d}  ||F||inf = {nrm:.3e}")
        if nrm < tol:
            break
        J = jacobian(g, x, R)
        dx = spla.spsolve(J.tocsc(), -F)
        if backtrack:
            lam = 1.0
            for _ in range(20):
                xn = x + lam*dx
                if np.linalg.norm(residual(g, xn, R), np.inf) < nrm:
                    break
                lam *= 0.5
            x = x + lam*dx
        else:
            x = x + dx
    return x, hist


def continuation(g, R_target, R_steps=None, verbose=False):
    if R_steps is None:
        R_steps = [r for r in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0] if r <= R_target]
        if R_target not in R_steps:
            R_steps.append(R_target)
    x = np.zeros(g.N)
    last_hist = []
    for R in R_steps:
        if verbose:
            print(f"  continuacion R = {R}")
        x, last_hist = newton(g, R, x0=x, verbose=verbose)
    return x, last_hist


# =====================================================================
if __name__ == "__main__":
    # malla gruesa (la de nuestro informe: 81x9 nodos, dominio 400x40, h=5)
    g = Grid(81, 9, obs1=(0, 6, 6, 8), obs2=(38, 42, 0, 4))
    print("Nodos totales:", g.Nx*g.Ny, " incognitas:", g.N)
    from collections import Counter
    print("Tipos:", Counter(g.type.ravel().tolist()))

    # ---- verificacion del Jacobiano vs diferencias finitas ----
    rng = np.random.default_rng(0)
    x = rng.standard_normal(g.N)*0.1
    R = 2.0
    J = jacobian(g, x, R).toarray()
    eps = 1e-6
    Jfd = np.zeros_like(J)
    F0 = residual(g, x, R)
    for c in range(g.N):
        xp = x.copy(); xp[c] += eps
        Jfd[:, c] = (residual(g, xp, R) - F0)/eps
    err = np.abs(J - Jfd).max()
    print(f"Verificacion Jacobiano  max|J_analitico - J_FD| = {err:.3e}")
