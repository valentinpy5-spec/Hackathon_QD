"""Couche limite intégrale, couplage faible (une seule passe, sans retour sur
l'écoulement non visqueux).

    laminaire   : Thwaites (quadrature) + corrélations H(lambda), l(lambda)
    transition  : critère de Michel, ou décollement laminaire, ou déclenchement forcé
    turbulent   : méthode d'entraînement de Head + frottement Ludwig-Tillmann
    traînée     : formule de Squire-Young au bord de fuite

Tout est adimensionné par c et U_inf : s en cordes, Ue = Ue/U_inf, et
nu = 1/Re.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

# Seuils de décollement
LAMBDA_SEP = -0.09   # Thwaites : décollement laminaire
H_SEP_TURB = 2.4     # Head : décollement turbulent (valeurs usuelles 1.8 à 2.8)
H_TURB_0 = 1.4       # facteur de forme initial après transition


# ---------------------------------------------------------------- laminaire
def thwaites_closure(lam):
    """Corrélations de Thwaites (forme Cebeci-Bradshaw) : H(lambda), l(lambda)."""
    lam = np.clip(lam, -0.1, 0.1)
    H = np.where(lam >= 0, 2.61 - 3.75 * lam + 5.24 * lam**2,
                 2.088 + 0.0731 / (lam + 0.14))
    ell = np.where(lam >= 0, 0.22 + 1.57 * lam - 1.8 * lam**2,
                   0.22 + 1.402 * lam + 0.018 * lam / (lam + 0.107))
    return H, ell


def michel_transition(Re_x):
    """Critère de Michel (1951) : transition si Re_theta dépasse cette valeur."""
    return 1.174 * (1 + 22400 / Re_x) * Re_x**0.46


# ---------------------------------------------------------------- turbulent
def H1_of_H(H):
    """Facteur de forme d'entraînement de Head, H1 = (delta - delta*)/theta."""
    H = np.asarray(H, dtype=float)
    return np.where(H <= 1.6, 3.3 + 0.8234 * (H - 1.1) ** -1.287,
                    3.3 + 1.5501 * (H - 0.6778) ** -3.064)


def H_of_H1(H1):
    H1 = np.maximum(np.asarray(H1, dtype=float), 3.3 + 1e-6)
    return np.where(H1 >= 5.3, 1.1 + ((H1 - 3.3) / 0.8234) ** (-1 / 1.287),
                    0.6778 + ((H1 - 3.3) / 1.5501) ** (-1 / 3.064))


def cf_ludwieg_tillmann(H, Re_theta):
    return 0.246 * 10 ** (-0.678 * H) * np.maximum(Re_theta, 1.0) ** -0.268


# ---------------------------------------------------------------- résultats
@dataclass
class SurfaceBL:
    """Couche limite le long d'une face (extrados ou intrados)."""
    s: np.ndarray          # abscisse curviligne depuis le point d'arrêt
    x: np.ndarray          # x/c des stations
    Ue: np.ndarray
    theta: np.ndarray
    H: np.ndarray
    cf: np.ndarray
    turbulent: np.ndarray  # booléen par station
    x_tr: float = np.nan   # x/c de la transition
    tr_mode: str = ""      # 'michel', 'laminar_sep', 'forced', 'none'
    x_sep: float = np.nan  # x/c du décollement turbulent (nan si attaché)
    flags: list = field(default_factory=list)

    @property
    def theta_te(self):
        return self.theta[-1]

    @property
    def H_te(self):
        return self.H[-1]

    @property
    def Ue_te(self):
        return self.Ue[-1]


def march_surface(s, x, Ue, Re, x_trip=None, x_stop=0.985, free_transition=True):
    """Résout la couche limite sur une face.

    s, x, Ue : stations, du point d'arrêt (exclu, s > 0) vers le bord de fuite.
    x_trip   : x/c de déclenchement forcé. Comme le « xtrip » de XFOIL, la
               transition a lieu au premier des deux : critère libre ou x_trip.
    free_transition : False désactive le critère de Michel (seuls x_trip et le
               décollement laminaire déclenchent) ; sert à caler sur XFOIL.
    x_stop   : x/c où l'on arrête la marche et applique Squire-Young. Un bord
               de fuite fermé est un point d'arrêt pour l'écoulement potentiel :
               Ue y chute vers 0 sur le dernier pour-cent de corde. Ce
               ralentissement n'existe pas dans l'écoulement réel (le sillage et
               l'épaisseur de déplacement l'effacent) et ferait exploser theta.
    """
    keep = x <= x_stop
    keep[: np.argmax(x > 0.5)] = True   # ne coupe jamais côté bord d'attaque
    s, x, Ue = s[keep], x[keep], Ue[keep]
    n = s.size
    flags = []
    Ue = np.maximum(Ue, 1e-6)
    dUe = np.gradient(Ue, s)

    # ---- Thwaites : theta^2 Ue^6 = 0.45 nu int_0^s Ue^5 ds (Ue(0) = 0)
    s0 = np.concatenate([[0.0], s])
    U0 = np.concatenate([[0.0], Ue])
    integ = np.concatenate([[0.0], np.cumsum(0.5 * (U0[1:]**5 + U0[:-1]**5) * np.diff(s0))])[1:]
    theta_lam = np.sqrt(0.45 / Re * integ / Ue**6)
    lam = theta_lam**2 * dUe * Re
    H_lam, ell = thwaites_closure(lam)
    Re_th = Ue * theta_lam * Re
    Re_x = np.maximum(Ue * s * Re, 1.0)

    # ---- recherche de la transition
    # Pour chaque critère, g >= 0 signifie « franchi ». La position exacte est
    # interpolée entre les deux stations qui encadrent le franchissement : sans
    # cela, x_tr saute de station en station et CD(m, p, t) devient en escalier,
    # ce qui dégrade le modèle quadratique de la phase 4.
    crit = {}
    if x_trip is not None:
        crit["forced"] = x - x_trip
    crit["laminar_sep"] = np.where(s > 0.002, LAMBDA_SEP - lam, -1.0)
    if free_transition:
        crit["michel"] = Re_th - michel_transition(Re_x)
    i_tr, mode, w = n, "none", 0.0
    for i in range(n):
        hit = [k for k, g in crit.items() if g[i] >= 0]
        if hit:
            i_tr, mode = i, hit[0]
            if i > 0:
                g0, g1 = crit[mode][i - 1], crit[mode][i]
                w = float(np.clip(-g0 / (g1 - g0), 0.0, 1.0)) if g1 != g0 else 1.0
            break

    theta = theta_lam.copy()
    H = H_lam.copy()
    cf = 2 * ell / np.maximum(Re_th, 1e-12)
    turb = np.zeros(n, dtype=bool)
    x_sep = np.nan

    if i_tr < n:
        # ---- Head, de i_tr jusqu'au bord de fuite
        st = s[i_tr:]
        if i_tr > 0:
            s_tr = s[i_tr - 1] + w * (s[i_tr] - s[i_tr - 1])
            th0 = float(np.interp(s_tr, s, theta_lam))
        else:
            s_tr, th0 = s[0], theta_lam[0]
        H1_0 = float(H1_of_H(H_TURB_0))
        H1_sep = float(H1_of_H(H_SEP_TURB))

        def Ue_f(si):
            return np.interp(si, s, Ue)

        def dUe_f(si):
            return np.interp(si, s, dUe)

        def rhs(si, y):
            th, E = y                       # E = Ue * theta * H1
            ue = Ue_f(si)
            H1 = max(E / (ue * th), H1_sep)  # au-delà du décollement : H gelé
            Hh = float(H_of_H1(H1))
            cft = float(cf_ludwieg_tillmann(Hh, ue * th * Re))
            dth = 0.5 * cft - (Hh + 2) * th / ue * dUe_f(si)
            dE = ue * 0.0306 * (H1 - 3.0) ** -0.6169
            return [dth, dE]

        y0 = [th0, Ue_f(s_tr) * th0 * H1_0]
        if st[-1] > s_tr:
            sol = solve_ivp(rhs, (s_tr, st[-1]), y0, t_eval=st,
                            method="RK45", rtol=1e-6, atol=1e-12)
            if not sol.success:
                flags.append("head_ivp_failed")
            th_t, E_t = sol.y
        else:
            th_t, E_t = np.array([y0[0]]), np.array([y0[1]])
        H1_t = E_t / (Ue[i_tr:i_tr + th_t.size] * th_t)
        H_t = H_of_H1(np.maximum(H1_t, H1_sep))
        theta[i_tr:i_tr + th_t.size] = th_t
        H[i_tr:i_tr + th_t.size] = H_t
        cf[i_tr:i_tr + th_t.size] = cf_ludwieg_tillmann(H_t, Ue[i_tr:i_tr + th_t.size] * th_t * Re)
        turb[i_tr:] = True
        sep = np.nonzero(H1_t <= H1_sep + 1e-9)[0]
        if sep.size:
            x_sep = x[i_tr + sep[0]]
            flags.append("turbulent_separation")
    else:
        if np.any(lam < LAMBDA_SEP):
            flags.append("laminar_separation_no_transition")

    if i_tr == n:
        x_tr = np.nan
    elif i_tr == 0:
        x_tr = x[0]
    else:
        x_tr = x[i_tr - 1] + w * (x[i_tr] - x[i_tr - 1])
    return SurfaceBL(s, x, Ue, theta, H, cf, turb, x_tr, mode, x_sep, flags)


def squire_young(theta_te, H_te, Ue_te):
    """Contribution d'une face à la traînée de profil : Cd = 2 theta (Ue/U)^((H+5)/2)."""
    return 2.0 * theta_te * Ue_te ** ((H_te + 5.0) / 2.0)


def split_at_stagnation(sol):
    """Découpe la surface en extrados / intrados à partir du point d'arrêt.

    Sur l'extrados, la tangente de panneau pointe vers le bord d'attaque : la
    vitesse tangentielle de l'écoulement réel y est négative. Le point d'arrêt
    est donc le changement de signe de Vt (de - vers +) en parcourant le contour.
    Renvoie deux tuples (s, x, Ue, idx) orientés du point d'arrêt vers le bord
    de fuite, idx étant l'indice des panneaux correspondants.
    """
    P, Vt = sol.panels, sol.Vt
    k = np.nonzero((Vt[:-1] < 0) & (Vt[1:] >= 0))[0]
    if k.size != 1:
        raise RuntimeError(f"point d'arrêt ambigu ({k.size} changements de signe)")
    k = k[0]
    # abscisse curviligne cumulée aux collocations
    s_nodes = np.concatenate([[0.0], np.cumsum(P.L)])
    s_c = 0.5 * (s_nodes[:-1] + s_nodes[1:])
    # interpolation linéaire du zéro de Vt entre k et k+1
    w = Vt[k] / (Vt[k] - Vt[k + 1])
    s_stag = s_c[k] + w * (s_c[k + 1] - s_c[k])

    up = np.arange(k, -1, -1)            # du point d'arrêt vers le BF extrados
    lo = np.arange(k + 1, P.n)           # du point d'arrêt vers le BF intrados
    upper = (s_stag - s_c[up], P.xc[up], -Vt[up], up)
    lower = (s_c[lo] - s_stag, P.xc[lo], Vt[lo], lo)
    return upper, lower
