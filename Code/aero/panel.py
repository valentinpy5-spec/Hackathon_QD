"""Méthode des panneaux de Hess-Smith (2D, incompressible, non visqueux).

Sur chaque panneau j : une source d'intensité constante sigma_j (inconnue
propre au panneau) et un tourbillon d'intensité gamma (une seule inconnue,
la même sur tous les panneaux). Soit N + 1 inconnues pour N panneaux :
    - N équations de non-pénétration  V . n_i = 0  aux points de collocation ;
    - 1 condition de Kutta : vitesses tangentielles égales en norme sur les
      deux panneaux du bord de fuite.

Adimensionnement : corde c = 1, U_inf = 1. Toutes les vitesses sont donc des
rapports U / U_inf et Cp = 1 - Vt^2.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Panels:
    xs: np.ndarray   # début de panneau
    ys: np.ndarray
    xe: np.ndarray   # fin de panneau
    ye: np.ndarray
    xc: np.ndarray   # point de collocation (milieu)
    yc: np.ndarray
    L: np.ndarray    # longueur
    tx: np.ndarray   # tangente unitaire (sens de parcours)
    ty: np.ndarray
    nx: np.ndarray   # normale sortante
    ny: np.ndarray

    @property
    def n(self):
        return self.L.size


def make_panels(X, Y):
    xs, ys, xe, ye = X[:-1], Y[:-1], X[1:], Y[1:]
    L = np.hypot(xe - xs, ye - ys)
    tx, ty = (xe - xs) / L, (ye - ys) / L
    return Panels(xs, ys, xe, ye, 0.5 * (xs + xe), 0.5 * (ys + ye),
                  L, tx, ty, ty.copy(), -tx.copy())


def influence_matrices(P):
    """Vitesses induites aux collocations i par des panneaux unitaires j.

    Renvoie (usx, usy, uvx, uvy) : composantes globales (x, y) de la vitesse
    induite par une source unitaire (s) et un tourbillon unitaire (v).

    Dans le repère local du panneau j (xi le long de t_j, eta le long de la
    normale gauche, qui pointe vers l'intérieur du profil) :
        source    : u = ln(r1/r2)/2pi,      v = (th2 - th1)/2pi
        tourbillon: u = -(th2 - th1)/2pi,   v = ln(r1/r2)/2pi
    (tourbillon positif = sens trigonométrique).
    """
    dx = P.xc[:, None] - P.xs[None, :]
    dy = P.yc[:, None] - P.ys[None, :]
    tx, ty, L = P.tx[None, :], P.ty[None, :], P.L[None, :]
    xi = dx * tx + dy * ty
    eta = -dx * ty + dy * tx
    r1 = np.hypot(xi, eta)
    r2 = np.hypot(xi - L, eta)
    lnr = np.log(r1 / r2)
    dth = np.arctan2(eta, xi - L) - np.arctan2(eta, xi)
    # Auto-influence : collocation sur le panneau, côté extérieur (eta -> 0-).
    idx = np.arange(P.n)
    lnr[idx, idx] = 0.0
    dth[idx, idx] = -np.pi

    us_l, vs_l = lnr / (2 * np.pi), dth / (2 * np.pi)
    uv_l, vv_l = -dth / (2 * np.pi), lnr / (2 * np.pi)
    # local -> global : e_xi = (tx, ty), e_eta = (-ty, tx)
    usx, usy = us_l * tx - vs_l * ty, us_l * ty + vs_l * tx
    uvx, uvy = uv_l * tx - vv_l * ty, uv_l * ty + vv_l * tx
    return usx, usy, uvx, uvy


@dataclass
class InviscidSolution:
    panels: Panels
    alpha: float        # rad
    sigma: np.ndarray
    gamma: float
    Vt: np.ndarray      # vitesse tangentielle signée (selon t_i) aux collocations
    Cp: np.ndarray
    CL: float
    CD_p: float         # traînée de pression intégrée : ~0 (d'Alembert), sert de contrôle
    CM: float           # moment de tangage au quart de corde, positif à cabrer
    CL_kj: float        # portance par Kutta-Joukowski (contrôle de cohérence)


def solve(X, Y, alpha_deg, cache=None):
    """Résout l'écoulement potentiel autour du profil (X, Y) à l'incidence alpha.

    cache : dict optionnel pour réutiliser les matrices d'influence lorsque
    seule l'incidence change (la géométrie, donc la matrice, est inchangée).
    """
    a = np.deg2rad(alpha_deg)
    if cache is not None and "P" in cache:
        P, (usx, usy, uvx, uvy), lu = cache["P"], cache["M"], cache["lu"]
    else:
        P = make_panels(X, Y)
        usx, usy, uvx, uvy = influence_matrices(P)
        lu = None
    N = P.n
    Uinf = np.array([np.cos(a), np.sin(a)])

    # Composantes normales et tangentielles des influences
    Asn = usx * P.nx[:, None] + usy * P.ny[:, None]
    Avn = uvx * P.nx[:, None] + uvy * P.ny[:, None]
    Ast = usx * P.tx[:, None] + usy * P.ty[:, None]
    Avt = uvx * P.tx[:, None] + uvy * P.ty[:, None]

    if lu is None:
        A = np.zeros((N + 1, N + 1))
        A[:N, :N] = Asn
        A[:N, N] = Avn.sum(axis=1)
        # Kutta : Vt_0 + Vt_{N-1} = 0 (tangentes opposées au bord de fuite)
        A[N, :N] = Ast[0] + Ast[-1]
        A[N, N] = Avt[0].sum() + Avt[-1].sum()
        lu = np.linalg.inv(A)  # petit système dense, réutilisé pour chaque alpha
        if cache is not None:
            cache.update(P=P, M=(usx, usy, uvx, uvy), lu=lu)

    b = np.empty(N + 1)
    b[:N] = -(Uinf[0] * P.nx + Uinf[1] * P.ny)
    b[N] = -(Uinf[0] * (P.tx[0] + P.tx[-1]) + Uinf[1] * (P.ty[0] + P.ty[-1]))
    q = lu @ b
    sigma, gamma = q[:N], q[N]

    Vt = Uinf[0] * P.tx + Uinf[1] * P.ty + Ast @ sigma + gamma * Avt.sum(axis=1)
    Cp = 1.0 - Vt**2

    # Efforts : dF = -Cp n dL (corde unitaire, pression dynamique unitaire)
    fx, fy = -Cp * P.nx * P.L, -Cp * P.ny * P.L
    Fx, Fy = fx.sum(), fy.sum()
    CL = Fy * np.cos(a) - Fx * np.sin(a)
    CD_p = Fx * np.cos(a) + Fy * np.sin(a)
    # Moment au quart de corde, positif à cabrer (bord d'attaque vers le haut)
    CM = -np.sum((P.xc - 0.25) * fy - P.yc * fx)
    # Circulation horaire = -gamma * périmètre ; CL = 2 Gamma / (U c)
    CL_kj = -2.0 * gamma * P.L.sum()
    return InviscidSolution(P, a, sigma, gamma, Vt, Cp, CL, CD_p, CM, CL_kj)
