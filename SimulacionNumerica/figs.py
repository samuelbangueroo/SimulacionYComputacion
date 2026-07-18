import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'figure.dpi': 110, 'savefig.dpi': 110, 'axes.linewidth': 0.6,
    'font.family': 'DejaVu Sans'
})

d = np.load('data.npz')
xf, yf = d['xf_ax'], d['yf_ax']
xc, yc = d['xc_ax'], d['yc_ax']
Pf, Of, vxf, vyf, spf = d['Pf'], d['Of'], d['vxf'], d['vyf'], d['spf']
Pc, Oc = d['Pc'], d['Oc']
Ps, Os, vxs, vys, sps = d['Ps'], d['Os'], d['vxs'], d['vys'], d['sps']
solid_f, solid_c = d['solid_f'], d['solid_c']
histf, histc = d['histf'], d['histc']
hj, hg, hcg, hgm = d['hj'], d['hg'], d['hcg'], d['hgm']

# --- helpers: los campos son [i=x, j=y]; para imshow uso .T con origin lower ---
def masked(F, solid):
    return np.ma.array(F.T, mask=solid.T)

extent_f = [xf[0], xf[-1], yf[0], yf[-1]]
extent_c = [xc[0], xc[-1], yc[0], yc[-1]]

def draw_obstacles(ax, solid, x, y):
    # dibuja el contorno de los solidos en gris
    ax.imshow(np.ma.array(np.ones_like(solid.T), mask=~solid.T),
              extent=[x[0], x[-1], y[0], y[-1]], origin='lower',
              cmap=mcolors.ListedColormap(['#4a4a4a']), aspect='auto', zorder=3)

def heat(ax, F, solid, x, y, extent, cmap, title, clip=None, sym=False):
    Fm = masked(F, solid)
    if clip is not None:
        vals = F[~solid]
        if sym:
            v = np.percentile(np.abs(vals), clip); vmin, vmax = -v, v
        else:
            vmin = np.percentile(vals, 100-clip); vmax = np.percentile(vals, clip)
    else:
        vmin, vmax = Fm.min(), Fm.max()
    im = ax.imshow(Fm, extent=extent, origin='lower', cmap=cmap,
                   aspect='auto', vmin=vmin, vmax=vmax, zorder=2)
    draw_obstacles(ax, solid, x, y)
    ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    return im

# ============================================================
# FIG 1: campos principales sobre malla fina (psi, omega, |v|)
# ============================================================
fig, axs = plt.subplots(3, 1, figsize=(7.2, 7.0), constrained_layout=True)
heat(axs[0], Pf, solid_f, xf, yf, extent_f, 'viridis',
     r'Funcion de corriente  $\psi$')
# lineas de corriente encima
Xc, Yc = np.meshgrid(xf, yf)
Pm = masked(Pf, solid_f)
axs[0].contour(Xc, Yc, Pm, levels=18, colors='white', linewidths=0.4, alpha=0.6, zorder=4)

heat(axs[1], Of, solid_f, xf, yf, extent_f, 'RdBu_r',
     r'Vorticidad  $\omega$', clip=99, sym=True)

heat(axs[2], spf, solid_f, xf, yf, extent_f, 'inferno',
     r'Rapidez  $|\mathbf{v}|$  (escala recortada al pct. 98)', clip=98)
# quiver ralo
step = 6
Xq, Yq = np.meshgrid(xf[::step], yf[::step])
U = vxf[::step, ::step].T; V = vyf[::step, ::step].T
Sm = solid_f[::step, ::step].T
U = np.ma.array(U, mask=Sm); V = np.ma.array(V, mask=Sm)
axs[2].quiver(Xq, Yq, U, V, color='white', scale=35, width=0.0025,
              alpha=0.8, zorder=5)
fig.savefig('fig1_campos.png', bbox_inches='tight')
plt.close(fig)

# ============================================================
# FIG 2: verificacion spline  (fina | gruesa+spline | error)  para psi y |v|
# ============================================================
fig, axs = plt.subplots(3, 2, figsize=(9.2, 7.2), constrained_layout=True)
# columna 0: psi
heat(axs[0,0], Pf, solid_f, xf, yf, extent_f, 'viridis', r'$\psi$ fina (referencia)')
heat(axs[1,0], Ps, solid_f, xf, yf, extent_f, 'viridis', r'$\psi$ gruesa + spline cubico')
heat(axs[2,0], np.abs(Pf-Ps), solid_f, xf, yf, extent_f, 'magma',
     r'$|\psi_{fina}-\psi_{spline}|$', clip=99)
# columna 1: |v|
heat(axs[0,1], spf, solid_f, xf, yf, extent_f, 'inferno', r'$|v|$ fina (referencia)', clip=98)
heat(axs[1,1], sps, solid_f, xf, yf, extent_f, 'inferno', r'$|v|$ gruesa + spline cubico', clip=98)
heat(axs[2,1], np.abs(spf-sps), solid_f, xf, yf, extent_f, 'magma',
     r'$|\,|v|_{fina}-|v|_{spline}|$', clip=99)
fig.savefig('fig2_spline.png', bbox_inches='tight')
plt.close(fig)

# ============================================================
# FIG 3: malla gruesa cruda vs spline (para ver que hace el spline)
# ============================================================
fig, axs = plt.subplots(2, 1, figsize=(7.2, 4.8), constrained_layout=True)
heat(axs[0], Pc, solid_c, xc, yc, extent_c, 'viridis',
     r'$\psi$ en malla gruesa ($81\times9$, $h=5$) — resuelta')
heat(axs[1], Ps, solid_f, xf, yf, extent_f, 'viridis',
     r'$\psi$ reconstruida a malla fina ($161\times17$, $h=2.5$) por spline')
fig.savefig('fig3_gruesa_fina.png', bbox_inches='tight')
plt.close(fig)

# ============================================================
# FIG 4: convergencia de Newton
# ============================================================
fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
ax.semilogy(range(1, len(histf)+1), histf, 'o-', label='malla fina', lw=1.4, ms=5)
ax.semilogy(range(1, len(histc)+1), histc, 's--', label='malla gruesa', lw=1.4, ms=5)
ax.set_xlabel('iteracion de Newton'); ax.set_ylabel(r'$\|F(x)\|_\infty$')
ax.set_title('Convergencia cuadratica de Newton-Raphson')
ax.grid(True, which='both', alpha=0.3); ax.legend()
fig.savefig('fig4_newton.png', bbox_inches='tight')
plt.close(fig)

# ============================================================
# FIG 5: convergencia de metodos para el sistema lineal J H = -F
# ============================================================
fig, ax = plt.subplots(figsize=(6.6, 4.0), constrained_layout=True)
def clean(h):  # recorta a residuo relativo
    h = np.asarray(h, float)
    return np.clip(h, 1e-12, 1e60)
ax.semilogy(range(1, len(hj)+1), clean(hj), label='Jacobi (diverge)', lw=1.3)
ax.semilogy(range(1, len(hg)+1), clean(hg), label='Gauss-Seidel (diverge)', lw=1.3)
ax.semilogy(range(1, len(hcg)+1), clean(hcg), label='Grad. Conjugado (estanca)', lw=1.0, alpha=0.8)
ax.semilogy(range(1, len(hgm)+1), clean(hgm), 'o-', label='GMRES + ILU (converge)', lw=1.6, ms=5)
ax.axhline(1e-8, color='k', ls=':', lw=0.8, label='tolerancia $10^{-8}$')
ax.set_xlabel('iteracion'); ax.set_ylabel('residuo relativo  $\\|b-Jx\\|/\\|b\\|$')
ax.set_title(r'Metodos iterativos para $J\,H=-F$ (Jacobiana no simetrica)')
ax.set_xlim(0, 80); ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=8, loc='center right')
fig.savefig('fig5_metodos.png', bbox_inches='tight')
plt.close(fig)

# ============================================================
# FIG 6: patron de dispersion de la Jacobiana
# ============================================================
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from solver import Grid, continuation, jacobian, residual
gL = Grid(81, 9, h=5.0, obs1=(0,6,6,8), obs2=(38,42,0,4))
x1,_ = continuation(gL, 1.0)
Jc = jacobian(gL, x1, 2.0).tocsr()
fig, ax = plt.subplots(figsize=(4.6, 4.6), constrained_layout=True)
ax.spy(Jc, markersize=0.5, color='#1f4e79')
ax.set_title('Estructura dispersa de la Jacobiana\n(%d$\\times$%d, nnz=%d, %.2f%% llena)'
             % (Jc.shape[0], Jc.shape[1], Jc.nnz, 100*Jc.nnz/Jc.shape[0]**2))
fig.savefig('fig6_jacobiana.png', bbox_inches='tight')
plt.close(fig)

import os
for f in ['fig1_campos.png','fig2_spline.png','fig3_gruesa_fina.png',
          'fig4_newton.png','fig5_metodos.png','fig6_jacobiana.png']:
    print(f, os.path.getsize(f)//1024, 'KB')
