"""
Spline cubico natural implementado desde cero (algoritmo de Thomas O(n))
y refinamiento 2D por producto tensorial.

Condicion natural:  S''(x0) = S''(xn) = 0.
Las incognitas son las segundas derivadas M_i = S''(x_i).
"""
import numpy as np


def natural_cubic_coeffs(x, y):
    """Devuelve M_i = S''(x_i) resolviendo el sistema tridiagonal (Thomas)."""
    n = len(x) - 1
    h = np.diff(x)
    M = np.zeros(n + 1)
    if n < 2:
        return M
    # sistema tridiagonal para M[1..n-1]
    a = np.zeros(n - 1)  # sub
    b = np.zeros(n - 1)  # diag
    c = np.zeros(n - 1)  # super
    d = np.zeros(n - 1)  # rhs
    for k in range(1, n):
        i = k - 1
        a[i] = h[k - 1]
        b[i] = 2.0 * (h[k - 1] + h[k])
        c[i] = h[k]
        d[i] = 6.0 * ((y[k + 1] - y[k]) / h[k] - (y[k] - y[k - 1]) / h[k - 1])
    # Thomas
    for i in range(1, n - 1):
        w = a[i] / b[i - 1]
        b[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]
    m = np.zeros(n - 1)
    m[-1] = d[-1] / b[-1]
    for i in range(n - 3, -1, -1):
        m[i] = (d[i] - c[i] * m[i + 1]) / b[i]
    M[1:n] = m
    return M


def spline_eval(x, y, M, xq):
    """Evalua el spline (con segundas derivadas M) en los puntos xq."""
    n = len(x) - 1
    h = np.diff(x)
    xq = np.atleast_1d(xq).astype(float)
    out = np.empty_like(xq)
    idx = np.clip(np.searchsorted(x, xq) - 1, 0, n - 1)
    for t, (xi, k) in enumerate(zip(xq, idx)):
        hk = h[k]
        A = (x[k + 1] - xi) / hk
        B = (xi - x[k]) / hk
        out[t] = (A * y[k] + B * y[k + 1]
                  + ((A**3 - A) * M[k] + (B**3 - B) * M[k + 1]) * hk**2 / 6.0)
    return out


def spline_1d(x, y, xq):
    return spline_eval(x, y, natural_cubic_coeffs(x, y), xq)


def refine_field(F_coarse, xc, yc, xf, yf):
    """
    Refina un campo 2D F_coarse (en malla xc x yc) a la malla xf x yf
    por producto tensorial de splines 1D: primero en x (por filas),
    luego en y (por columnas).
    F_coarse tiene forma (len(xc), len(yc)).
    """
    nxc, nyc = F_coarse.shape
    # 1) interpolar en x cada fila (indice y fijo)  -> (len(xf), nyc)
    tmp = np.empty((len(xf), nyc))
    for jj in range(nyc):
        tmp[:, jj] = spline_1d(xc, F_coarse[:, jj], xf)
    # 2) interpolar en y cada columna (indice x fino fijo) -> (len(xf), len(yf))
    out = np.empty((len(xf), len(yf)))
    for ii in range(len(xf)):
        out[ii, :] = spline_1d(yc, tmp[ii, :], yf)
    return out


# ---------------------------------------------------------------------
if __name__ == "__main__":
    # test 1: interpolacion exacta en los nodos
    x = np.linspace(0, 10, 11)
    y = np.sin(x)
    M = natural_cubic_coeffs(x, y)
    assert np.allclose(spline_eval(x, y, M, x), y, atol=1e-12)
    # test 2: condicion natural M0=Mn=0
    assert abs(M[0]) < 1e-14 and abs(M[-1]) < 1e-14
    # test 3: comparar con scipy
    try:
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(x, y, bc_type="natural")
        xq = np.linspace(0, 10, 97)
        err = np.abs(cs(xq) - spline_1d(x, y, xq)).max()
        print("max discrepancia vs scipy CubicSpline(natural):", err)
        assert err < 1e-10
    except ImportError:
        pass
    print("spline: todos los tests OK")
