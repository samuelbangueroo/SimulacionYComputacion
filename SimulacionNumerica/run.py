import numpy as np, time, json
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from solver import Grid, continuation, newton, reconstruct, residual, jacobian
from spline import refine_field

R = 2.0
V0 = 1.0

# ---------- geometrias equivalentes (FACTOR 2) ----------
FINE = dict(Nx=161, Ny=17, h=2.5, obs1=(0,12,12,16), obs2=(76,84,0,8))
COARSE = dict(Nx=81, Ny=9,  h=5.0, obs1=(0,6,6,8),    obs2=(38,42,0,4))

def velocities(Psi, h):
    vx = np.zeros_like(Psi); vy = np.zeros_like(Psi)
    vx[:,1:-1] = (Psi[:,2:]-Psi[:,:-2])/(2*h)
    vy[1:-1,:] = -(Psi[2:,:]-Psi[:-2,:])/(2*h)
    vx[:,0]=vx[:,1]; vx[:,-1]=vx[:,-2]; vy[0,:]=vy[1,:]; vy[-1,:]=vy[-2,:]
    return vx, vy, np.sqrt(vx**2+vy**2)

# ============ 1) SOLUCION FINA (referencia) ============
gf = Grid(FINE['Nx'],FINE['Ny'],h=FINE['h'],obs1=FINE['obs1'],obs2=FINE['obs2'])
t=time.time(); xf,histf = continuation(gf,R); tf=time.time()-t
Pf,Of = reconstruct(gf,xf)
vxf,vyf,spf = velocities(Pf,gf.h)

# ============ 2) SOLUCION GRUESA ============
gc = Grid(COARSE['Nx'],COARSE['Ny'],h=COARSE['h'],obs1=COARSE['obs1'],obs2=COARSE['obs2'])
t=time.time(); xc,histc = continuation(gc,R); tc=time.time()-t
Pc,Oc = reconstruct(gc,xc)

# ============ 3) RECONSTRUCCION POR SPLINE (gruesa -> fina) ============
xc_ax = np.arange(gc.Nx)*gc.h;  yc_ax = np.arange(gc.Ny)*gc.h
xf_ax = np.arange(gf.Nx)*gf.h;  yf_ax = np.arange(gf.Ny)*gf.h
Ps = refine_field(Pc, xc_ax, yc_ax, xf_ax, yf_ax)
Os = refine_field(Oc, xc_ax, yc_ax, xf_ax, yf_ax)
vxs,vys,sps = velocities(Ps,gf.h)

# ============ 4) METRICAS DE ERROR (spline vs fina directa) ============
mask = ~gf.solid
def stats(a,b):
    e=np.abs(a-b)[mask]
    return dict(mediana=float(np.median(e)), p90=float(np.percentile(e,90)),
                maxv=float(e.max()))
err_psi=stats(Ps,Pf); err_om=stats(Os,Of); err_sp=stats(sps,spf)

# verificacion: el spline es exacto en los nodos gruesos coincidentes
coincide = np.abs(Ps[::2,::2]-Pc).max()

# ============ 5) EFECTO DEL TERMINO NO LINEAL (bajo Reynolds) ============
# comparamos la solucion completa (R=2) contra la de Stokes (R=0, sin conveccion)
g0 = Grid(FINE['Nx'],FINE['Ny'],h=FINE['h'],obs1=FINE['obs1'],obs2=FINE['obs2'])
x0,_ = newton(g0,0.0)             # R=0 -> sistema lineal (dos Poisson acoplados)
P0,O0 = reconstruct(g0,x0)
rel_psi = float(np.linalg.norm((Pf-P0)[mask])/np.linalg.norm(Pf[mask]))
vx0,vy0,sp0 = velocities(P0,gf.h)
rel_v = float(np.linalg.norm(np.r_[(vxf-vx0)[mask],(vyf-vy0)[mask]])
              /np.linalg.norm(np.r_[vxf[mask],vyf[mask]]))
cellRe = float(R*spf[mask].mean())
nl_summary = dict(rel_change_psi=rel_psi, rel_change_v=rel_v, cellRe_mean=cellRe)

# ============ 6) COMPARACION DE METODOS PARA EL SISTEMA LINEAL J H = -F ============
# congelamos un paso de Newton real: solucion a R=1, evaluando la fisica de R=2
gL = Grid(COARSE['Nx'],COARSE['Ny'],h=COARSE['h'],obs1=COARSE['obs1'],obs2=COARSE['obs2'])
x1,_ = continuation(gL,1.0)
Jc = jacobian(gL,x1,R).tocsr()
bc = -residual(gL,x1,R)
n = Jc.shape[0]
xexact = spla.spsolve(Jc.tocsc(),bc)

def relres(x): return np.linalg.norm(bc-Jc@x)/np.linalg.norm(bc)

# --- Directo (LU dispersa) ---
t=time.time(); xd=spla.spsolve(Jc.tocsc(),bc); td=time.time()-t
methods={'Directo (LU dispersa)':dict(conv=True,iters=None,t=td,err=0.0)}

# --- Jacobi ---
D=Jc.diagonal(); LU=Jc-sp.diags(D)
def jacobi(maxit=5000,tol=1e-8):
    x=np.zeros(n); hist=[]
    for k in range(maxit):
        x=(bc-LU@x)/D; r=relres(x); hist.append(r)
        if r<tol or r>1e50: break
    return x,hist
xj,hj=jacobi()
methods['Jacobi']=dict(conv=bool(hj[-1]<1e-8),iters=len(hj),t=None,
                       err=float(np.linalg.norm(xj-xexact)/np.linalg.norm(xexact)),hist=hj)

# --- Gauss-Seidel ---
L=sp.tril(Jc).tocsr(); U=(Jc-L).tocsr()
def gauss_seidel(maxit=5000,tol=1e-8):
    x=np.zeros(n); hist=[]
    for k in range(maxit):
        x=spla.spsolve_triangular(L,bc-U@x,lower=True); r=relres(x); hist.append(r)
        if r<tol or r>1e50: break
    return x,hist
t=time.time(); xg,hg=gauss_seidel(); tg=time.time()-t
methods['Gauss-Seidel']=dict(conv=bool(hg[-1]<1e-8),iters=len(hg),t=tg,
                       err=float(np.linalg.norm(xg-xexact)/np.linalg.norm(xexact)),hist=hg)

# --- Gradiente Conjugado (requiere SPD; J no lo es) ---
cg_hist=[]
def cb(xk): cg_hist.append(relres(xk))
try:
    xcg,info=spla.cg(Jc,bc,rtol=1e-8,maxiter=5000,callback=cb)
except TypeError:
    xcg,info=spla.cg(Jc,bc,tol=1e-8,maxiter=5000,callback=cb)
methods['Gradiente Conjugado']=dict(conv=bool(info==0),iters=len(cg_hist),t=None,
                       err=float(np.linalg.norm(xcg-xexact)/np.linalg.norm(xexact)),hist=cg_hist)

# --- GMRES con precondicionador ILU (Krylov para no simetricos) ---
ilu=spla.spilu(Jc.tocsc()); Mx=spla.LinearOperator(Jc.shape, ilu.solve)
gm_hist=[]
def cbg(rk): gm_hist.append(float(rk))
try:
    xgm,infg=spla.gmres(Jc,bc,rtol=1e-8,maxiter=500,M=Mx,callback=cbg,callback_type='pr_norm')
except TypeError:
    xgm,infg=spla.gmres(Jc,bc,tol=1e-8,maxiter=500,M=Mx,callback=cbg)
t=time.time(); spla.gmres(Jc,bc,rtol=1e-8,maxiter=500,M=Mx) if True else None; tgm=time.time()-t
methods['GMRES + ILU']=dict(conv=bool(infg==0),iters=len(gm_hist),t=tgm,
                       err=float(np.linalg.norm(xgm-xexact)/np.linalg.norm(xexact)),hist=gm_hist)

sym_err = float(abs(Jc-Jc.T).max())

# ============ guardar todo ============
np.savez('data.npz',
    xf_ax=xf_ax,yf_ax=yf_ax,xc_ax=xc_ax,yc_ax=yc_ax,
    Pf=Pf,Of=Of,vxf=vxf,vyf=vyf,spf=spf,
    Pc=Pc,Oc=Oc,
    Ps=Ps,Os=Os,vxs=vxs,vys=vys,sps=sps,
    solid_f=gf.solid, solid_c=gc.solid,
    histf=np.array(histf),histc=np.array(histc),
    hj=np.array(hj),hg=np.array(hg),hcg=np.array(cg_hist),hgm=np.array(gm_hist),
    P0=P0,O0=O0,sp0=sp0)

summary=dict(
    R=R,
    fine=dict(N=int(gf.N),iters=len(histf),res=float(histf[-1]),t=float(tf),
              nodes=int(gf.Nx*gf.Ny)),
    coarse=dict(N=int(gc.N),iters=len(histc),res=float(histc[-1]),t=float(tc),
                nodes=int(gc.Nx*gc.Ny)),
    err_psi=err_psi,err_om=err_om,err_sp=err_sp,
    spline_node_match=float(coincide),
    psi_top=float(V0*(gf.Ny-1)*gf.h),
    speed_max_fine=float(spf[mask].max()),
    speed_p98_fine=float(np.percentile(spf[mask],98)),
    nonlin=nl_summary,
    Jshape=int(n), Jnnz=int(Jc.nnz), density=float(Jc.nnz/n**2),
    sym_err=sym_err,
    linres_frozen=float(np.linalg.norm(bc,np.inf)),
    methods={k:{kk:vv for kk,vv in v.items() if kk!='hist'} for k,v in methods.items()},
)
json.dump(summary,open('summary.json','w'),indent=2)
print(json.dumps(summary,indent=2))
