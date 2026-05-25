# =============================================================================
# Solución de Navier-Stokes estacionario e incompresible (formulación u-w)
# Flujo en canal 2D con dos obstáculos rectangulares
# Método: Newton-Raphson con Jacobiano numérico por perturbación
# Librerías permitidas: numpy, matplotlib.pyplot, mpl_toolkits.mplot3d
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D  # importada según requisitos (no se usa directamente)

# =============================================================================
# PARÁMETROS GLOBALES DEL PROBLEMA
# =============================================================================

L   = 400       # Longitud del canal [m]
H   = 40        # Altura del canal [m]
nu  = 1.0       # Viscosidad cinemática
dx  = 5.0       # Paso de malla en x
dy  = 5.0       # Paso de malla en y
Nx  = int(L / dx) + 1   # 81 nodos en x (índices 0..80)
Ny  = int(H / dy) + 1   # 9  nodos en y (índices 0..8)

# Valor de u en las paredes horizontal
U_FONDO = 0.0     # u = 0 en y = 0 (pared inferior)
U_TAPA  = 40.0    # u = 40 en y = 40 (pared superior)

# Obstáculo 1: bloque adosado a la tapa (esquina superior izquierda)
# x ∈ [0,30], y ∈ [30,40]  → u = 40 (adosado a la tapa)
OBS1_x0, OBS1_x1 = 0,   30
OBS1_y0, OBS1_y1 = 30,  40
U_OBS1 = U_TAPA

# Obstáculo 2: viga central adosada al fondo
# x ∈ [190,210], y ∈ [0,30]  → u = 0 (adosado al fondo)
OBS2_x0, OBS2_x1 = 190, 210
OBS2_y0, OBS2_y1 = 0,   30
U_OBS2 = U_FONDO

# =============================================================================
# 1. CREAR MALLA
# =============================================================================

def crear_malla():
    """
    Crea las coordenadas de la malla uniforme.
    Retorna:
        x  : array 1D de coordenadas x (Nx,)
        y  : array 1D de coordenadas y (Ny,)
        X  : malla 2D de x (Nx, Ny)
        Y  : malla 2D de y (Nx, Ny)
        mascara_obs : array booleano (Nx, Ny), True donde hay obstáculo sólido
    """
    x = np.linspace(0.0, L, Nx)   # x: 0, 5, 10, ..., 400
    y = np.linspace(0.0, H, Ny)   # y: 0, 5, 10, ..., 40

    X, Y = np.meshgrid(x, y, indexing='ij')  # X,Y con forma (Nx, Ny)

    # Máscara de obstáculos: True si el nodo está dentro de un sólido
    mascara_obs = np.zeros((Nx, Ny), dtype=bool)

    for i in range(Nx):
        for j in range(Ny):
            xi = x[i]
            yj = y[j]
            # Bloque superior (adosado a la tapa)
            if OBS1_x0 <= xi <= OBS1_x1 and OBS1_y0 <= yj <= OBS1_y1:
                mascara_obs[i, j] = True
            # Viga central (adosada al fondo)
            if OBS2_x0 <= xi <= OBS2_x1 and OBS2_y0 <= yj <= OBS2_y1:
                mascara_obs[i, j] = True

    return x, y, X, Y, mascara_obs


# =============================================================================
# 2. APLICAR CONDICIONES DE BORDE
# =============================================================================

def aplicar_condiciones_borde(u, w, mascara_obs, y):
    """
    Impone las condiciones de contorno sobre u y w:
      - Entrada (i=0):    u = y,  w = 0
      - Salida  (i=Nx-1): extrapolación de primer orden (∂/∂x = 0)
      - Fondo   (j=0):    u = 0,  w de Thom
      - Tapa    (j=Ny-1): u = 40, w de Thom
      - Obstáculos:       u fijo, w de Thom en caras expuestas

    Parámetros:
        u, w        : arrays (Nx, Ny) actuales
        mascara_obs : array booleano (Nx, Ny)
        y           : coordenadas y (Ny,)
    Retorna:
        u, w modificados in-place
    """

    # --- Entrada (i = 0): perfil uniforme u_vel=1 → u = y, w = 0 ---
    for j in range(Ny):
        u[0, j]   = y[j]      # u(0,j) = y_j
        w[0, j] = 0.0       # sin vorticidad en la entrada

    # --- Salida (i = Nx-1): condición de gradiente nulo (extrapolación) ---
    for j in range(Ny):
        u[Nx-1, j]   = u[Nx-2, j]
        w[Nx-1, j] = w[Nx-2, j]

    # --- Pared fondo (j = 0): no deslizamiento, u = 0 ---
    for i in range(Nx):
        u[i, 0] = U_FONDO
        # Fórmula de Thom: w_fondo = -2*(u[i,1] - u[i,0]) / dy²
        w[i, 0] = -2.0 * (u[i, 1] - u[i, 0]) / (dy * dy)

    # --- Pared tapa (j = Ny-1): no deslizamiento, u = 40 ---
    for i in range(Nx):
        u[i, Ny-1] = U_TAPA
        # Fórmula de Thom: w_tapa = -2*(u[i,Ny-2] - u[i,Ny-1]) / dy²
        w[i, Ny-1] = -2.0 * (u[i, Ny-2] - u[i, Ny-1]) / (dy * dy)

    # --- Obstáculos: fijar u y calcular w de Thom en nodos sólidos ---
    for i in range(Nx):
        for j in range(Ny):
            if mascara_obs[i, j]:
                xi = i * dx
                yj = j * dy

                # Bloque superior (adosado a la tapa): u = 40
                if OBS1_x0 <= xi <= OBS1_x1 and OBS1_y0 <= yj <= OBS1_y1:
                    u[i, j] = U_OBS1

                # Viga central (adosada al fondo): u = 0
                if OBS2_x0 <= xi <= OBS2_x1 and OBS2_y0 <= yj <= OBS2_y1:
                    u[i, j] = U_OBS2

    # Calcular w de Thom en las caras expuestas de los obstáculos
    # (nodos sólidos adyacentes a nodos fluidos en dirección normal)
    for i in range(Nx):
        for j in range(Ny):
            if mascara_obs[i, j]:
                # Cara inferior del obstáculo (nodo fluido por debajo: j-1)
                if j > 0 and not mascara_obs[i, j-1]:
                    # Cara inferior sólida: equivalente a "pared superior"
                    # El nodo fluido está abajo → w en sólido como tapa
                    w[i, j] = -2.0 * (u[i, j-1] - u[i, j]) / (dy * dy)
                # Cara superior del obstáculo (nodo fluido por encima: j+1)
                elif j < Ny-1 and not mascara_obs[i, j+1]:
                    # Cara superior sólida: equivalente a "pared inferior"
                    w[i, j] = -2.0 * (u[i, j+1] - u[i, j]) / (dy * dy)
                # Cara derecha del obstáculo (nodo fluido a la derecha: i+1)
                elif i < Nx-1 and not mascara_obs[i+1, j]:
                    w[i, j] = -2.0 * (u[i+1, j] - u[i, j]) / (dx * dx)
                # Cara izquierda del obstáculo (nodo fluido a la izquierda: i-1)
                elif i > 0 and not mascara_obs[i-1, j]:
                    w[i, j] = -2.0 * (u[i-1, j] - u[i, j]) / (dx * dx)

    return u, w


# =============================================================================
# 3. DETERMINAR NODOS LIBRES (incógnitas de Newton-Raphson)
# =============================================================================

def obtener_nodos_libres(mascara_obs):
    """
    Identifica los nodos fluidos interiores cuyas variables son incógnitas
    del sistema de Newton-Raphson. Se excluyen:
      - Borde de entrada (i=0)
      - Borde de salida  (i=Nx-1)
      - Pared fondo      (j=0)
      - Pared tapa       (j=Ny-1)
      - Nodos de obstáculos

    Retorna:
        nodos_libres : lista de tuplas (i, j) de nodos libres
        idx_map      : dict {(i,j): índice en el vector X} para u
                       el índice para w del mismo nodo es idx+N_libres
    """
    nodos_libres = []
    for i in range(1, Nx-1):       # excluye entrada y salida
        for j in range(1, Ny-1):   # excluye paredes
            if not mascara_obs[i, j]:
                nodos_libres.append((i, j))

    idx_map = {nodo: k for k, nodo in enumerate(nodos_libres)}
    return nodos_libres, idx_map


# =============================================================================
# 4. CALCULAR RESIDUOS
# =============================================================================

def calcular_residuos(u, w, nodos_libres, mascara_obs):
    """
    Calcula el vector de residuos F para todos los nodos libres.
    Primer bloque (índices 0..N-1):  residuos de la ecuación de Poisson para u
    Segundo bloque (índices N..2N-1): residuos de la ecuación de vorticidad

    F_u(i,j)  = u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] + dx²·w[i,j] - 4·u[i,j]
    F_w(i,j) = u_vel·(w[i+1,j]-w[i-1,j])/(2dx) + v_vel·(w[i,j+1]-w[i,j-1])/(2dy)
                  - ν·((w[i+1,j]-2w[i,j]+w[i-1,j])/dx² + (w[i,j+1]-2w[i,j]+w[i,j-1])/dy²)

    Las velocidades se calculan localmente:
        u_vel[i,j] = (u[i,j+1] - u[i,j-1]) / (2dy)
        v_vel[i,j] = -(u[i+1,j] - u[i-1,j]) / (2dx)

    Parámetros:
        u, w        : arrays (Nx, Ny)
        nodos_libres: lista de (i,j)
        mascara_obs : array booleano (Nx, Ny)
    Retorna:
        F : vector numpy (2*N_libres,)
    """
    N = len(nodos_libres)
    F = np.zeros(2 * N)

    for k, (i, j) in enumerate(nodos_libres):
        # Vecinos necesarios (todos existen porque i ∈ [1,Nx-2] y j ∈ [1,Ny-2])
        u_ip = u[i+1, j]
        u_im = u[i-1, j]
        u_jp = u[i, j+1]
        u_jm = u[i, j-1]
        u_c  = u[i, j]

        w_ip = w[i+1, j]
        w_im = w[i-1, j]
        w_jp = w[i, j+1]
        w_jm = w[i, j-1]
        w_c  = w[i, j]

        # Velocidades por diferencias centradas a partir de u
        u_vel_ij = (u_jp - u_jm) / (2.0 * dy)
        v_vel_ij = -(u_ip - u_im) / (2.0 * dx)

        # --- Residuo de Poisson (u) ---
        F_u = (u_ip + u_im + u_jp + u_jm
                 + dx * dx * w_c
                 - 4.0 * u_c)
        F[k] = F_u

        # --- Residuo de transporte de vorticidad (w) ---
        adv_x  = u_vel_ij * (w_ip - w_im) / (2.0 * dx)
        adv_y  = v_vel_ij * (w_jp - w_jm) / (2.0 * dy)
        dif_x  = (w_ip - 2.0 * w_c + w_im) / (dx * dx)
        dif_y  = (w_jp - 2.0 * w_c + w_jm) / (dy * dy)
        F_w = adv_x + adv_y - nu * (dif_x + dif_y)
        F[N + k] = F_w

    return F


# =============================================================================
# 5. CALCULAR JACOBIANO POR PERTURBACIÓN NUMÉRICA
# =============================================================================

def calcular_jacobiano(u, w, nodos_libres, idx_map, mascara_obs, y, F0):
    """
    Calcula la matriz Jacobiana J de tamaño (2N x 2N) usando diferencias finitas
    hacia adelante con perturbación ε = 1e-8.

    J[k, l] = (F_k(X + ε·e_l) - F_k(X)) / ε

    donde el vector X contiene primero los u de los nodos libres y luego los w.

    Parámetros:
        u, w        : arrays (Nx, Ny) con la solución actual
        nodos_libres: lista de (i,j)
        idx_map     : dict {(i,j): índice k}
        mascara_obs : array booleano
        y           : coordenadas y
        F0          : residuo F en el punto actual (sin perturbar)
    Retorna:
        J : matriz numpy (2N, 2N)
    """
    N   = len(nodos_libres)
    eps = 1.0e-8
    J   = np.zeros((2 * N, 2 * N))

    for l in range(2 * N):
        # Perturbar la variable l-ésima
        u_p   = u.copy()
        w_p = w.copy()

        if l < N:
            # Perturbación en u del nodo l
            i_l, j_l = nodos_libres[l]
            u_p[i_l, j_l] += eps
        else:
            # Perturbación en w del nodo (l-N)
            i_l, j_l = nodos_libres[l - N]
            w_p[i_l, j_l] += eps

        # Aplicar condiciones de borde con los arrays perturbados
        u_p, w_p = aplicar_condiciones_borde(u_p, w_p, mascara_obs, y)

        # Calcular residuos perturbados
        F_p = calcular_residuos(u_p, w_p, nodos_libres, mascara_obs)

        # Columna l del Jacobiano
        J[:, l] = (F_p - F0) / eps

    return J


# =============================================================================
# 6. INICIALIZAR u Y w
# =============================================================================

def inicializar_campos(y, mascara_obs):
    """
    Condición inicial: flujo potencial uniforme.
      u(x, y) = y  (perfil lineal)
      w(x, y) = 0  en nodos fluidos
    Luego se imponen condiciones de borde.

    Retorna u, w arrays (Nx, Ny)
    """
    u   = np.zeros((Nx, Ny))
    w = np.zeros((Nx, Ny))

    # u = y en todo el dominio como condición inicial
    for j in range(Ny):
        u[:, j] = y[j]

    # Fijar valores de u en obstáculos
    for i in range(Nx):
        for j in range(Ny):
            xi = i * dx
            yj = j * dy
            if OBS1_x0 <= xi <= OBS1_x1 and OBS1_y0 <= yj <= OBS1_y1:
                u[i, j] = U_OBS1
            if OBS2_x0 <= xi <= OBS2_x1 and OBS2_y0 <= yj <= OBS2_y1:
                u[i, j] = U_OBS2

    return u, w


# =============================================================================
# 7. NEWTON-RAPHSON
# =============================================================================

def newton_solve():
    """
    Resuelve el sistema acoplado (u, w) mediante Newton-Raphson.

    Algoritmo:
      1. Crear malla y máscara de obstáculos.
      2. Inicializar u y w.
      3. Aplicar condiciones de borde.
      4. Repetir hasta convergencia (||F||_inf < 1e-8) o 50 iteraciones:
         a. Calcular residuos F0.
         b. Calcular Jacobiano J.
         c. Resolver J·ΔX = -F0.
         d. Actualizar nodos libres con ΔX.
         e. Aplicar condiciones de borde.
      5. Reportar número de iteraciones y norma residual.

    Retorna:
        u, w       : solución convergida (Nx, Ny)
        mascara_obs: máscara de obstáculos (Nx, Ny)
        X, Y       : mallas de coordenadas (Nx, Ny)
    """
    # --- Crear malla ---
    x, y, X, Y, mascara_obs = crear_malla()

    # --- Obtener nodos libres ---
    nodos_libres, idx_map = obtener_nodos_libres(mascara_obs)
    N = len(nodos_libres)
    print(f"Número de nodos libres: {N}  →  Sistema de {2*N} incógnitas")

    # --- Inicializar campos ---
    u, w = inicializar_campos(y, mascara_obs)

    # --- Aplicar condiciones de borde iniciales ---
    u, w = aplicar_condiciones_borde(u, w, mascara_obs, y)

    tol      = 1.0e-8
    max_iter = 50

    print("\nIteración   ||F||_inf")
    print("-" * 30)

    for iteracion in range(1, max_iter + 1):

        # --- Calcular residuos en el punto actual ---
        F0     = calcular_residuos(u, w, nodos_libres, mascara_obs)
        norma  = np.max(np.abs(F0))

        print(f"  {iteracion:3d}       {norma:.6e}")

        # --- Verificar convergencia ---
        if norma < tol:
            print(f"\n✓ Convergido en {iteracion} iteraciones. ||F||_inf = {norma:.3e}")
            break

        # --- Calcular Jacobiano por perturbación numérica ---
        J = calcular_jacobiano(u, w, nodos_libres, idx_map,
                               mascara_obs, y, F0)

        # --- Resolver sistema lineal J·ΔX = -F0 ---
        try:
            delta_X = np.linalg.solve(J, -F0)
        except np.linalg.LinAlgError:
            print("  ¡Jacobiano singular! Abortando.")
            break

        # --- Actualizar nodos libres ---
        for k, (i, j) in enumerate(nodos_libres):
            u[i, j]   += delta_X[k]
            w[i, j] += delta_X[N + k]

        # --- Aplicar condiciones de borde tras actualización ---
        u, w = aplicar_condiciones_borde(u, w, mascara_obs, y)

    else:
        # Se alcanzó el máximo de iteraciones sin convergir
        F_final = calcular_residuos(u, w, nodos_libres, mascara_obs)
        norma_final = np.max(np.abs(F_final))
        print(f"\n⚠ Máximo de iteraciones ({max_iter}) alcanzado. ||F||_inf = {norma_final:.3e}")

    return u, w, mascara_obs, X, Y


# =============================================================================
# 8. CALCULAR VELOCIDAD
# =============================================================================

def calcular_velocidad(u):
    """
    Calcula las componentes de velocidad mediante diferencias centradas:
        u_vel[i,j] = (u[i,j+1] - u[i,j-1]) / (2·Δy)
        v_vel[i,j] = -(u[i+1,j] - u[i-1,j]) / (2·Δx)

    En los bordes se usa diferencia unilateral (hacia adelante o atrás).

    Retorna:
        u_vel, v_vel : arrays (Nx, Ny)
    """
    u_vel = np.zeros((Nx, Ny))
    v_vel = np.zeros((Nx, Ny))

    # Nodos interiores: diferencias centradas
    u_vel[1:Nx-1, 1:Ny-1] = (u[1:Nx-1, 2:Ny] - u[1:Nx-1, 0:Ny-2]) / (2.0 * dy)
    v_vel[1:Nx-1, 1:Ny-1] = -(u[2:Nx, 1:Ny-1] - u[0:Nx-2, 1:Ny-1]) / (2.0 * dx)

    # Borde izquierdo (entrada, i=0): diferencia hacia adelante en x
    u_vel[0, 1:Ny-1] = (u[0, 2:Ny] - u[0, 0:Ny-2]) / (2.0 * dy)
    v_vel[0, 1:Ny-1] = -(u[1, 1:Ny-1] - u[0, 1:Ny-1]) / dx

    # Borde derecho (salida, i=Nx-1): diferencia hacia atrás en x
    u_vel[Nx-1, 1:Ny-1] = (u[Nx-1, 2:Ny] - u[Nx-1, 0:Ny-2]) / (2.0 * dy)
    v_vel[Nx-1, 1:Ny-1] = -(u[Nx-1, 1:Ny-1] - u[Nx-2, 1:Ny-1]) / dx

    # Bordes superior e inferior (diferencias unilaterales en y)
    u_vel[1:Nx-1, 0]    = (u[1:Nx-1, 1] - u[1:Nx-1, 0])    / dy
    u_vel[1:Nx-1, Ny-1] = (u[1:Nx-1, Ny-1] - u[1:Nx-1, Ny-2]) / dy

    return u_vel, v_vel


# =============================================================================
# 9. GRAFICAR RESULTADOS
# =============================================================================

def graficar_resultados(u, w, u_vel, v_vel, mascara_obs, X, Y):
    """
    Genera dos figuras:
      1. Mapa de calor de la magnitud de velocidad |V| = sqrt(u_vel²+v_vel²)
         usando pcolormesh con colormap 'jet'.
      2. Líneas de corriente con contour de u (20-30 niveles)
         con los obstáculos superpuestos como rectángulos.

    Parámetros:
        u          : función corriente (Nx, Ny)
        u_vel, v_vel : componentes de velocidad (Nx, Ny)
        mascara_obs: máscara booleana de obstáculos (Nx, Ny)
        X, Y       : mallas de coordenadas (Nx, Ny)
    """

    # Magnitud de velocidad
    Vmag = np.sqrt(u_vel**2 + v_vel**2)

    # Enmascarar interior de obstáculos para la visualización
    Vmag_plot = Vmag.copy().astype(float)
    Vmag_plot[mascara_obs] = np.nan

    u_plot = u.copy().astype(float)
    u_plot[mascara_obs] = np.nan

    # =========================================================================
    # Figura 1: Mapa de calor de magnitud de velocidad
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(14, 4))

    # pcolormesh espera X e Y de los nodos (o de las celdas +1)
    # Usamos X.T e Y.T porque X,Y tienen indexing='ij' → forma (Nx,Ny)
    pcm = ax1.pcolormesh(X.T, Y.T, Vmag_plot.T,
                         cmap='jet',
                         shading='auto',
                         vmin=0.0,
                         vmax=np.nanmax(Vmag_plot))

    # Añadir los obstáculos como rectángulos sólidos grises
    rect1 = patches.Rectangle((OBS1_x0, OBS1_y0),
                               OBS1_x1 - OBS1_x0,
                               OBS1_y1 - OBS1_y0,
                               linewidth=1.5,
                               edgecolor='black',
                               facecolor='gray',
                               zorder=5)
    rect2 = patches.Rectangle((OBS2_x0, OBS2_y0),
                               OBS2_x1 - OBS2_x0,
                               OBS2_y1 - OBS2_y0,
                               linewidth=1.5,
                               edgecolor='black',
                               facecolor='gray',
                               zorder=5)
    ax1.add_patch(rect1)
    ax1.add_patch(rect2)

    cbar1 = fig1.colorbar(pcm, ax=ax1)
    cbar1.set_label('|V| [m/s]', fontsize=11)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Magnitud de velocidad', fontsize=14)
    ax1.set_xlim(0, L)
    ax1.set_ylim(0, H)
    ax1.set_aspect('equal')
    fig1.tight_layout()

    # =========================================================================
    # Figura 2: Líneas de corriente (contornos de u)
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(14, 4))

    # Niveles de contorno entre u_fondo y u_tapa
    niveles = np.linspace(U_FONDO, U_TAPA, 25)

    cs = ax2.contour(X.T, Y.T, u_plot.T,
                     levels=niveles,
                     cmap='viridis',
                     linewidths=1.0)
    ax2.clabel(cs, inline=False, fontsize=6)

    # Añadir los obstáculos como rectángulos sólidos
    rect3 = patches.Rectangle((OBS1_x0, OBS1_y0),
                               OBS1_x1 - OBS1_x0,
                               OBS1_y1 - OBS1_y0,
                               linewidth=1.5,
                               edgecolor='black',
                               facecolor='dimgray',
                               zorder=5)
    rect4 = patches.Rectangle((OBS2_x0, OBS2_y0),
                               OBS2_x1 - OBS2_x0,
                               OBS2_y1 - OBS2_y0,
                               linewidth=1.5,
                               edgecolor='black',
                               facecolor='dimgray',
                               zorder=5)
    ax2.add_patch(rect3)
    ax2.add_patch(rect4)

    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_title('Líneas de corriente', fontsize=14)
    ax2.set_xlim(0, L)
    ax2.set_ylim(0, H)
    ax2.set_aspect('equal')
    fig2.tight_layout()

    # =========================================================================
    # Figura 3: Vorticidad w
    # =========================================================================
    w_plot = w.copy().astype(float)
    w_plot[mascara_obs] = np.nan   # ocultar obstáculos

    fig3, ax3 = plt.subplots(figsize=(14, 4))
    # Usar coolwarm porque w puede ser positiva o negativa
    pcm3 = ax3.pcolormesh(X.T, Y.T, w_plot.T,
                          cmap='coolwarm', shading='auto')
    cbar3 = fig3.colorbar(pcm3, ax=ax3)
    cbar3.set_label('w [1/s]', fontsize=11)
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('y', fontsize=12)
    ax3.set_title('Vorticidad', fontsize=14)
    ax3.set_xlim(0, L)
    ax3.set_ylim(0, H)
    ax3.set_aspect('equal')

    # Dibujar los obstáculos como rectángulos (igual que en las otras figuras)
    rect5 = patches.Rectangle((OBS1_x0, OBS1_y0),
                               OBS1_x1 - OBS1_x0,
                               OBS1_y1 - OBS1_y0,
                               linewidth=1.5, edgecolor='black', facecolor='gray', zorder=5)
    rect6 = patches.Rectangle((OBS2_x0, OBS2_y0),
                               OBS2_x1 - OBS2_x0,
                               OBS2_y1 - OBS2_y0,
                               linewidth=1.5, edgecolor='black', facecolor='gray', zorder=5)
    ax3.add_patch(rect5)
    ax3.add_patch(rect6)

    fig3.tight_layout()
    plt.show()


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

if __name__ == "__main__":

    print("=" * 50)
    print("  Navier-Stokes estacionario — Canal 2D con obstáculos")
    print("  Formulación u-w  |  Newton-Raphson")
    print("=" * 50)
    print(f"\nMalla: Nx={Nx}, Ny={Ny}  (Δx=Δy={dx})")
    print(f"Dominio: L={L}, H={H},  ν={nu}")
    print(f"Obstáculo 1 (tapa):  x∈[{OBS1_x0},{OBS1_x1}], y∈[{OBS1_y0},{OBS1_y1}]")
    print(f"Obstáculo 2 (fondo): x∈[{OBS2_x0},{OBS2_x1}], y∈[{OBS2_y0},{OBS2_y1}]")
    print()

    # --- Resolver el sistema ---
    u, w, mascara_obs, X, Y = newton_solve()

    # --- Calcular velocidades ---
    u_vel, v_vel = calcular_velocidad(u)

    # --- Graficar resultados ---
    graficar_resultados(u, w, u_vel, v_vel, mascara_obs, X, Y)
